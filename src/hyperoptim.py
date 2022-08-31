import os
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_absolute_percentage_error

from src import utils
from src.build_features import BuildFeatures

from functools import partial

import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback

config_path = "config.yaml"

config = utils.read_params(config_path)

feature_dir = config["build_features"]["feature_dir"]
feature_train = config["build_features"]["train"]

mlflow_config = config["mlflow_config"]
artifacts_dir = mlflow_config["artifacts_dir"] 
experiment_name = mlflow_config["experiment_name"]
tracking_uri = mlflow_config["remote_server_uri"]

mlflow.set_experiment(experiment_name)
mlflow.set_tracking_uri(tracking_uri)

mlflc = MLflowCallback(
    tracking_uri=tracking_uri,
    metric_name="mean_absolute_percentage_error",
)

@mlflc.track_in_mlflow()
def optimize(trial):
    """Optimize the model parameters"""
    n_components = trial.suggest_int('n_components', 6, 14)
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    solver = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
    max_iter = trial.suggest_int('max_iter', 1000, 3000)

    df = pd.read_csv(os.path.join("./", feature_dir, feature_train))

    features = list(df.columns)
    features.remove("Target")
    features.remove("kfold")

    model = linear_model.Ridge(alpha=alpha, solver=solver, max_iter=max_iter, random_state=42)

    maes = []

    for fold_num in range(5):
        xtrain = df[df["kfold"] != fold_num][features]
        ytrain = df[df["kfold"] != fold_num]["Target"]

        xvalid = df[df["kfold"] == fold_num][features]
        yvalid = df[df["kfold"] == fold_num]["Target"]

        xtrain_ft, xvalid_ft = BuildFeatures(config_path).build_features_for_model(xtrain, xvalid, None, n_components)
        
        model.fit(xtrain_ft, ytrain)
        preds = model.predict(xvalid_ft)
        mae = mean_absolute_percentage_error(yvalid, preds)
        maes.append(mae) 

    mae = np.mean(maes)
    
    mlflow.log_param("n_components", n_components)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("solver", solver)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("mean_absolute_percentage_error", mae)

    return np.mean(mae)

if __name__ == "__main__":
    opimization_function = partial(optimize)

    study = optuna.create_study(study_name = experiment_name, direction="minimize")
    study.optimize(opimization_function, n_trials=50, callbacks=[mlflc])