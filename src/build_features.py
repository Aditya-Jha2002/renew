# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
from src import utils
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
import logging
from dotenv import find_dotenv, load_dotenv

class BuildFeatures:
    """BuildFeatures class to take in train and test features and perform feature engineering"""

    def __init__(self, config_path):
        config = utils.read_params(config_path)
        self.clean_dir = config["make_dataset"]["clean_dir"]
        self.train_clean = config["make_dataset"]["train"]
        self.test_clean = config["make_dataset"]["test"]
        
        self.feature_dir = config["build_features"]["feature_dir"]
        self.train_features = config["build_features"]["train"]
        self.test_features = config["build_features"]["test"]
            
    def build_features(input_filepath, output_filepath):
        """ Runs feature engineering scripts to turn clean data from (../interim) into
            feature data ready to be modeled on (saved in ../processed).
        """
        logger = logging.getLogger(__name__)
        logger.info('making feature data set from clean data')

        # Load the cleaned dataset
        train = pd.read_csv(os.path.join(input_filepath, "train_clean.csv"))
        test = pd.read_csv(os.path.join(input_filepath, "test_clean.csv"))

        # new features
        # TBD

        # Save the features dataset
        train.to_csv(os.path.join(output_filepath, "train_features.csv"), index=False)
        test.to_csv(os.path.join(output_filepath, "test_features.csv"), index=False)

    def build_features_for_model(self, xtrain, xvalid, xtest, n_components):
        """ Runs feature engineering scripts to turn clean data from (../interim) into
            feature data ready to be modeled on (saved in ../processed).
        """
        logger = logging.getLogger(__name__)
        logger.info('making feature data set from clean data')

        # Seperate the features between categorical and numerical
        categorical_features = xtrain.select_dtypes(include=['object']).columns
        numerical_features = xtrain.select_dtypes(exclude=['object']).columns

        # Scale the numerical features
        scaler = RobustScaler()
        xtrain[numerical_features] = scaler.fit_transform(xtrain[numerical_features])
        xvalid[numerical_features] = scaler.transform(xvalid[numerical_features])
        if xtest is not None:
            xtest[numerical_features] = scaler.transform(xtest[numerical_features])

        # PCA on the features
        pca = PCA(n_components=n_components)
        xtrain_numerical = pca.fit_transform(xtrain[numerical_features])
        xvalid_numerical = pca.transform(xvalid[numerical_features])
        if xtest is not None:
            xtest_numerical = pca.transform(xtest[numerical_features])

        # One-hot encode the categorical features
        encoder = OneHotEncoder(sparse=False)
        xtrain_categorical = encoder.fit_transform(xtrain[categorical_features])
        xvalid_categorical = encoder.transform(xvalid[categorical_features])
        if xtest is not None:
            xtest_categorical = encoder.transform(xtest[categorical_features])

        # Concatenate the numerical and categorical features
        xtrain_features = np.column_stack([xtrain_numerical, xtrain_categorical])
        xvalid_features = np.column_stack([xvalid_numerical, xvalid_categorical])
        if xtest is not None:
            xtest_features = np.column_stack([xtest_numerical, xtest_categorical])

        # Return the features dataset
        if xtest is not None:
            return xtrain_features, xvalid_features, xtest_features
        else:
            return xtrain_features, xvalid_features

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config.yaml")
    parsed_args = args.parse_args()

    BuildFeatures(config_path=parsed_args.config).build_features()