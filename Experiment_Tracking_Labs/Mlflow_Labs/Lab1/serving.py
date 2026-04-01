"""
This example demonstrates how to specify pip requirements using `pip_requirements` and
`extra_pip_requirements` when logging a model via `mlflow.*.log_model`.
Dataset: Breast Cancer (sklearn built-in)
"""

import os
import tempfile

import sklearn
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.models.signature import infer_signature


def read_lines(path):
    with open(path) as f:
        return f.read().splitlines()


def get_pip_requirements(run_id, artifact_path, return_constraints=False):
    req_path = download_artifacts(run_id=run_id, artifact_path=f"{artifact_path}/requirements.txt")
    reqs = read_lines(req_path)

    if return_constraints:
        con_path = download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/constraints.txt"
        )
        cons = read_lines(con_path)
        return set(reqs), set(cons)

    return set(reqs)


def main():
    data = load_breast_cancer()
    dtrain = xgb.DMatrix(data.data, data.target)
    model = xgb.train({}, dtrain)
    predictions = model.predict(dtrain)
    signature = infer_signature(dtrain.get_data(), predictions)

    xgb_req = f"xgboost=={xgb.__version__}"
    sklearn_req = f"scikit-learn=={sklearn.__version__}"

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # 1. Default - auto-detects dependencies
        artifact_path = "default"
        mlflow.xgboost.log_model(model, artifact_path, signature=signature)
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert xgb_req in pip_reqs, pip_reqs
        print(f"1. Default requirements: {pip_reqs}")

        # 2. Override with pip_requirements
        artifact_path = "pip_requirements"
        mlflow.xgboost.log_model(
            model, artifact_path, pip_requirements=[sklearn_req], signature=signature
        )
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert sklearn_req in pip_reqs, pip_reqs
        print(f"2. Overridden requirements: {pip_reqs}")

        # 3. Add extra_pip_requirements on top of defaults
        artifact_path = "extra_pip_requirements"
        mlflow.xgboost.log_model(
            model, artifact_path, extra_pip_requirements=[sklearn_req], signature=signature
        )
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert pip_reqs.issuperset({xgb_req, sklearn_req}), pip_reqs
        print(f"3. Extra requirements: {pip_reqs}")

        # 4. Requirements from a file path (Windows-safe: delete=False)
        req_file = tempfile.NamedTemporaryFile("w", suffix=".requirements.txt", delete=False)
        try:
            req_file.write(sklearn_req)
            req_file.close()

            artifact_path = "requirements_file_path"
            mlflow.xgboost.log_model(
                model, artifact_path, pip_requirements=req_file.name, signature=signature
            )
            pip_reqs = get_pip_requirements(run_id, artifact_path)
            assert sklearn_req in pip_reqs, pip_reqs
            print(f"4. File path requirements: {pip_reqs}")

            # 5. Mix of direct strings + file reference
            artifact_path = "requirements_file_list"
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                pip_requirements=[xgb_req, f"-r {req_file.name}"],
                signature=signature,
            )
            pip_reqs = get_pip_requirements(run_id, artifact_path)
            assert pip_reqs.issuperset({xgb_req, sklearn_req}), pip_reqs
            print(f"5. File list requirements: {pip_reqs}")
        finally:
            os.unlink(req_file.name)

        # 6. Using a constraints file (Windows-safe: delete=False)
        con_file = tempfile.NamedTemporaryFile("w", suffix=".constraints.txt", delete=False)
        try:
            con_file.write(sklearn_req)
            con_file.close()

            artifact_path = "constraints_file"
            mlflow.xgboost.log_model(
                model,
                artifact_path,
                pip_requirements=[xgb_req, f"-c {con_file.name}"],
                signature=signature,
            )
            pip_reqs, pip_cons = get_pip_requirements(
                run_id, artifact_path, return_constraints=True
            )
            assert pip_reqs.issuperset({xgb_req, "-c constraints.txt"}), pip_reqs
            assert pip_cons == {sklearn_req}, pip_cons
            print(f"6. Constraints: reqs={pip_reqs}, cons={pip_cons}")
        finally:
            os.unlink(con_file.name)

    print("\nAll 6 requirement methods verified successfully!")


if __name__ == "__main__":
    main()