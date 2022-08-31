import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
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
    n_components = trial.suggest_int("n_components", 9, 14)
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_features = trial.suggest_int("max_features", 1, 9)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    df = pd.read_csv(os.path.join("./", feature_dir, feature_train))

    features = list(df.columns)
    features.remove("Target")
    features.remove("kfold")

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap, random_state=42)

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
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("max_features", max_features)
    mlflow.log_metric("mean_absolute_percentage_error", mae)

    return np.mean(mae)

if __name__ == "__main__":
    opimization_function = partial(optimize)

    study = optuna.create_study(study_name = experiment_name, direction="minimize")
    study.optimize(opimization_function, n_trials=150, callbacks=[mlflc])