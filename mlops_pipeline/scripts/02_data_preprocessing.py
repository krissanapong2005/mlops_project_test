import os, json
from yaml import safe_load
import mlflow

import os, mlflow  # (ให้อยู่บรรทัดบนๆ)
# ถ้ารันบน GitHub Actions ให้ใช้ local file backend เสมอ
if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true":
    mlflow.set_tracking_uri("file:./mlruns")

# อ่าน config + ตั้งชื่อ experiment (ให้รองรับ ENV บน CI)
cfg = safe_load(open("mlops_pipeline/config/params.yaml", encoding="utf-8"))
exp_name = os.getenv("EXPERIMENT_NAME", cfg["mlflow"]["experiment"])
mlflow.set_experiment(exp_name)

TRANSFORM = {
    "resize": [32, 32],
    "normalize_mean": [0.4914, 0.4822, 0.4465],
    "normalize_std":  [0.2470, 0.2435, 0.2616]
}

def main():
    cfg = safe_load(open("mlops_pipeline/config/params.yaml", encoding="utf-8"))
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run(run_name="preprocess_setup") as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "preprocessing")

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/transform.json", "w") as f:
            json.dump(TRANSFORM, f, indent=2)
        mlflow.log_artifact("artifacts/transform.json")

        print("transform.json logged")
        # ส่ง run_id ออกไปให้ workflow ตัวถัดไปใช้
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)
        else:
            print("Preprocessing run_id:", run_id)

if __name__ == "__main__":
    main()
