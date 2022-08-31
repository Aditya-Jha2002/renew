# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from src import utils
from src.build_features import BuildFeatures
import argparse
import joblib
import json
import logging
from dotenv import find_dotenv, load_dotenv


class Trainer:
    """Trains a model on the given dataset"""

    def __init__(self, config_path):
        self.config_path = config_path

        config = utils.read_params(config_path)

        self.submission_dir = config["base"]["submission_dir"]
        self.submission_file = config["base"]["submission_file"]

        self.raw_dir = config["raw_dataset"]["raw_dir"]     
        self.submission = config["raw_dataset"]["submission"]

        self.feature_dir = config["build_features"]["feature_dir"]
        self.train_features = config["build_features"]["train"]
        self.test_features = config["build_features"]["test"]

        self.cat_col = config["cols"]["cat_col"]
        self.cont_col = config["cols"]["cont_col"]
        self.target_col = config["cols"]["target_col"]

        self.n_components = config["estimators"]["DecisionTreeRegression"]["params"]["n_components"]
        self.n_estimators = config["estimators"]["DecisionTreeRegression"]["params"]["n_estimators"]
        self.max_features = config["estimators"]["DecisionTreeRegression"]["params"]["max_features"]
        self.max_depth = config["estimators"]["DecisionTreeRegression"]["params"]["max_depth"]
        self.min_samples_split = config["estimators"]["DecisionTreeRegression"]["params"]["min_samples_split"]
        self.min_samples_leaf = config["estimators"]["DecisionTreeRegression"]["params"]["min_samples_leaf"]
        self.bootstrap = config["estimators"]["DecisionTreeRegression"]["params"]["bootstrap"]
        self.random_state = config["base"]["random_state"]

        self.model_dir = config["model_dir"]

        self.report_dir = config["reports"]["report_dir"]
        self.params_file = config["reports"]["params_file"]
        self.scores_file = config["reports"]["scores_file"]

    def train_and_evaluate(self):
        """Train the model and evaluate the model performance"""
        logger = logging.getLogger(__name__)
        logger.info('creating model, reports and submission files')
        submission = pd.read_csv(os.path.join(self.raw_dir, self.submission))

        running_mae = []
        running_test_preds = np.array(len(submission) * [0.0])

        for i in range(1, 6):
            test_preds, mae = self._train_one_fold(i - 1)

            running_mae.append(float(mae))

            running_test_preds = np.column_stack((running_test_preds ,test_preds))
           
        running_mae = np.sum(running_mae)/5
        running_test_preds = np.sum(running_test_preds, axis=1)/5

        submission["Target"] = list(running_test_preds)

        submission.to_csv(os.path.join(self.submission_dir, self.submission_file), index=False)
        
        print("-" * 50)
        print("Ridge Regression Model")
        print(f"MAE: {running_mae}")
        print("-" * 50)
        print("-" * 50)

    #####################################################
    # Log Parameters and Scores for the deployed modle
    #####################################################
        with open(os.path.join(self.report_dir, self.scores_file), "w") as f:
            scores = {
                "mae_score": running_mae
                }
            json.dump(scores, f, indent=4)

        with open(os.path.join(self.report_dir, self.params_file), "w") as f:
            params = {
                "n_components": self.n_components,
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "bootstrap": self.bootstrap,
            }
            json.dump(params, f, indent=4)
    #####################################################

    def _train_one_fold(self, fold_num):
        print(f"Training fold {fold_num} ...")
        train_features = pd.read_csv(os.path.join(self.feature_dir, self.train_features))
        test_features = pd.read_csv(os.path.join(self.feature_dir, self.test_features))

        features = list(train_features.columns)
        features.remove("Target")
        features.remove("kfold")

        xtrain = train_features[train_features["kfold"] != fold_num][features]
        ytrain = train_features[train_features["kfold"] != fold_num]["Target"]

        xvalid = train_features[train_features["kfold"] == fold_num][features]
        yvalid = train_features[train_features["kfold"] == fold_num]["Target"]

        xtest = test_features[features]

        xtrain_ft, xvalid_ft, xtest_ft = BuildFeatures(self.config_path).build_features_for_model(xtrain, xvalid, xtest, self.n_components)

        clf = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, bootstrap=self.bootstrap, random_state=self.random_state)

        clf.fit(xtrain_ft, ytrain)
        
        valid_preds = clf.predict(xvalid_ft)
        mae = self._eval_metrics(yvalid, valid_preds)

        test_preds = clf.predict(xtest_ft)

        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"model_{fold_num}.joblib")

        joblib.dump(clf, model_path)

        print("-" * 50)
        print(f"  Fold {fold_num} score:")
        print(f"  Mean Absolute Percentage Error: {mae}")
        print("-" * 50)

        return test_preds, mae

    def _eval_metrics(self, actual, pred):
        """ Takes in the ground truth labels, predictions labels, and prediction probabilities.
            Returns the accuracy, f1, auc_roc, log_loss scores.
        """
        mae = mean_absolute_percentage_error(actual, pred)

        return mae

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config.yaml")
    parsed_args = args.parse_args()
    Trainer(config_path=parsed_args.config).train_and_evaluate()