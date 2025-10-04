import os, json
from yaml import safe_load
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
        client.create_experiment(exp_name, artifact_location="file:" + track_dir)
    mlflow.set_experiment(exp_name)
else:
    cfg_top = safe_load(open("mlops_pipeline/config/params.yaml", encoding="utf-8"))
    mlflow.set_experiment(cfg_top["mlflow"]["experiment"])
# ==================================

TRANSFORM = {
    "resize": [32, 32],
    "normalize_mean": [0.4914, 0.4822, 0.4465],
    "normalize_std":  [0.2470, 0.2435, 0.2616]
}

def main():
    with mlflow.start_run(run_name="preprocess_setup") as run:
        run_id = run.info.run_id
        mlflow.set_tag("ml.step", "preprocessing")

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/transform.json", "w") as f:
            json.dump(TRANSFORM, f, indent=2)
        mlflow.log_artifact("artifacts/transform.json")

        print("transform.json logged")
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)
        else:
            print("Preprocessing run_id:", run_id)

if __name__ == "__main__":
    main()
