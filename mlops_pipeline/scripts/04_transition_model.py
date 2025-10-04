import sys
from mlflow.tracking import MlflowClient

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/04_transition_model.py <model_name> <alias_or_stage>")
        sys.exit(1)

    name, target = sys.argv[1], sys.argv[2]
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        print(f"No versions for {name}")
        sys.exit(1)
    latest = max(versions, key=lambda v: int(v.version))
    try:
        client.set_registered_model_alias(name=name, alias=target, version=latest.version)
        print(f"Set alias '{target}' for {name} v{latest.version}")
    except Exception:
        client.transition_model_version_stage(name=name, version=latest.version, stage=target, archive_existing=True)
        print(f"Transitioned {name} v{latest.version} -> stage {target}")

if __name__ == "__main__":
    main()
