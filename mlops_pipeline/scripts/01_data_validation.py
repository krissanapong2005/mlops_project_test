import os
from collections import Counter
from yaml import safe_load
from torchvision import datasets, transforms
import mlflow
from mlflow.tracking import MlflowClient

# ==== CI-safe MLflow bootstrap ====
if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true":
    track_dir = os.path.abspath("./mlruns")
    mlflow.set_tracking_uri("file:" + track_dir)
    exp_name = os.getenv("EXPERIMENT_NAME", "cifar10-ci")
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = client.create_experiment(exp_name, artifact_location="file:" + track_dir)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(exp_name)
else:
    cfg_top = safe_load(open("mlops_pipeline/config/params.yaml", encoding="utf-8"))
    mlflow.set_experiment(cfg_top["mlflow"]["experiment"])
# ==================================

def main():
    cfg = safe_load(open("mlops_pipeline/config/params.yaml", encoding="utf-8"))
    root = cfg["dataset"]["root"]

    tfm = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    labels = train_ds.targets
    dist = Counter(labels)

    with mlflow.start_run(run_name="data_validation"):
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.log_metric("train_size", len(train_ds))
        mlflow.log_metric("test_size", len(test_ds))
        mlflow.log_param("num_classes", len(train_ds.classes))
        for k, v in dist.items():
            mlflow.log_metric(f"class_{k}_count", v)

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/data_report.txt", "w", encoding="utf-8") as f:
            f.write(f"Train: {len(train_ds)}\nTest: {len(test_ds)}\n")
            f.write("Classes: " + ", ".join(train_ds.classes) + "\n")
            for k, v in sorted(dist.items()):
                f.write(f"class_{k} ({train_ds.classes[k]}): {v}\n")
        mlflow.log_artifact("artifacts/data_report.txt")
        print("Data validation done.")

if __name__ == "__main__":
    main()
