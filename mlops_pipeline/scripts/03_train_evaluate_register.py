import os, sys, json, time, mlflow, torch
from yaml import safe_load
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from mlflow.tracking import MlflowClient

def build_transforms(spec):
    return transforms.Compose([
        transforms.Resize(tuple(spec["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(tuple(spec["normalize_mean"]), tuple(spec["normalize_std"]))
    ])

def load_transform_artifact(preprocess_run_id):
    try:
        path = mlflow.artifacts.download_artifacts(run_id=preprocess_run_id, artifact_path="transform.json")
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        # fallback hard-coded
        return {
            "resize": [32,32],
            "normalize_mean": [0.4914,0.4822,0.4465],
            "normalize_std":  [0.2470,0.2435,0.2616]
        }

def get_loaders(cfg, tfm):
    root = cfg["dataset"]["root"]
    full_train = datasets.CIFAR10(root=root, train=True,  download=False, transform=tfm)
    test_ds    = datasets.CIFAR10(root=root, train=False, download=False, transform= tfm)

    N = len(full_train)
    val_sz = int(0.10 * N)
    train_sz = N - val_sz
    train_ds, val_ds = random_split(full_train, [train_sz, val_sz], generator=torch.Generator().manual_seed(42))

    fast_env = os.getenv("FAST_DEV", "")
    fast = cfg["dataset"]["fast_dev"] or fast_env == "1"
    if fast:
        g = torch.Generator().manual_seed(123)
        tr_k = min(cfg["dataset"]["train_subset"], len(train_ds))
        va_k = min(cfg["dataset"]["val_subset"],  len(val_ds))
        train_ds = Subset(train_ds, torch.randperm(len(train_ds), generator=g)[:tr_k])
        val_ds   = Subset(val_ds,   torch.randperm(len(val_ds),   generator=g)[:va_k])

    nw = 0 if os.name == "nt" else max(0, cfg["train"]["num_workers"])
    bs = cfg["train"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw),
        test_ds.classes
    ), fast

def build_model():
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    return m

def evaluate(model, loader, device):
    model.eval(); tot=corr=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            corr += (out.argmax(1)==y).sum().item()
            tot  += y.size(0)
    return corr/tot

def main():
    if len(sys.argv) < 2:
        print("Usage: python 03_train_evaluate_register.py <preprocess_run_id> [threshold]")
        sys.exit(1)

    preprocess_run_id = sys.argv[1]
    custom_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None

    cfg = safe_load(open("mlops_pipeline/config/params.yaml", encoding="utf-8"))
    threshold = custom_threshold if custom_threshold is not None else cfg["eval"]["threshold"]

    tfm_spec = load_transform_artifact(preprocess_run_id)
    tfm = build_transforms(tfm_spec)

    (train_loader, val_loader, test_loader, classes), fast = get_loaders(cfg, tfm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    run_name = f'{cfg["train"]["model_name"]}_DEV' if fast else cfg["train"]["model_name"]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("ml.step", "train_eval_register")
        mlflow.log_param("epochs", cfg["train"]["epochs"])
        mlflow.log_param("batch_size", cfg["train"]["batch_size"])
        mlflow.log_param("lr", cfg["train"]["lr"])
        mlflow.log_param("fast_dev", fast)
        mlflow.log_param("classes", ",".join(classes))

        for epoch in range(cfg["train"]["epochs"]):
            model.train(); tot=corr=loss_sum=0.0
            for x,y in train_loader:
                x,y = x.to(device), y.to(device)
                opt.zero_grad(); out = model(x)
                loss = criterion(out,y); loss.backward(); opt.step()
                loss_sum += loss.item()*y.size(0)
                corr += (out.argmax(1)==y).sum().item()
                tot  += y.size(0)
            tr_acc, tr_loss = corr/tot, loss_sum/tot

            va_acc = evaluate(model, val_loader, device)
            mlflow.log_metric("train_acc", tr_acc, step=epoch)
            mlflow.log_metric("val_acc",   va_acc, step=epoch)
            print(f"Epoch {epoch+1}/{cfg['train']['epochs']}  tr_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")

        test_acc = evaluate(model, test_loader, device)
        mlflow.log_metric("test_acc", test_acc)
        print("Test acc:", test_acc)

        # log model
        mlflow.pytorch.log_model(model, "model")

        # try register (skip if registry not available)
        model_uri = f"runs:/{run.info.run_id}/model"
        model_name = cfg["mlflow"]["registered_model_name"]
        try:
            mv = mlflow.register_model(model_uri, model_name)
            client = MlflowClient()
            stage = "Production" if test_acc >= threshold else "Staging"
            # ถ้าใช้ Stages:
            try:
                client.transition_model_version_stage(name=model_name, version=mv.version, stage=stage, archive_existing=True)
            except Exception:
                # ถ้าใช้ Aliases:
                alias = "prod" if stage == "Production" else "staging"
                client.set_registered_model_alias(name=model_name, alias=alias, version=mv.version)
            print(f"Registered {model_name} v{mv.version} -> {stage}")
        except Exception as e:
            print(f"[WARN] Model Registry not available. Skipped registering. ({e})")

if __name__ == "__main__":
    main()
