import os
from collections import Counter
from yaml import safe_load
from torchvision import datasets, transforms
import mlflow
import os, mlflow  # (ให้อยู่บรรทัดบนๆ)
# ถ้ารันบน GitHub Actions ให้ใช้ local file backend เสมอ
if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true":
    mlflow.set_tracking_uri("file:./mlruns")

# อ่าน config + ตั้งชื่อ experiment (ให้รองรับ ENV บน CI)
cfg = safe_load(open("mlops_pipeline/config/params.yaml", encoding="utf-8"))
exp_name = os.getenv("EXPERIMENT_NAME", cfg["mlflow"]["experiment"])
mlflow.set_experiment(exp_name)

def main():
    cfg = safe_load(open("mlops_pipeline/config/params.yaml", encoding="utf-8"))
    root = cfg["dataset"]["root"]

    tfm = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)

    labels = train_ds.targets
    dist = Counter(labels)

    mlflow.set_experiment(cfg["mlflow"]["experiment"])
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
