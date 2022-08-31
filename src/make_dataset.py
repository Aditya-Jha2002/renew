# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
from src import utils
from sklearn import model_selection
import logging
from dotenv import find_dotenv, load_dotenv

class DataCleaner:
    """DataCleaner class to load the data, and preprocess it"""

    def __init__(self, config_path):
        config = utils.read_params(config_path)
        self.raw_dir = config["raw_dataset"]["raw_dir"]
        self.train_raw = config["raw_dataset"]["train"]
        self.test_raw = config["raw_dataset"]["test"]

        self.clean_dir = config["make_dataset"]["clean_dir"]
        self.train_clean = config["make_dataset"]["train"]
        self.test_clean = config["make_dataset"]["test"]
        
        self.fold_num = config["make_dataset"]["fold_num"]

    def clean_dataset(self):
        """ Runs data processing scripts to turn raw data from (../raw) into
            cleaned data ready to be feature engineered on (saved in ../interim).
        """
        logger = logging.getLogger(__name__)
        logger.info('making clean data set from raw data')

        # Load the raw dataset
        train = pd.read_csv(os.path.join(self.raw_dir, self.train_raw))
        test = pd.read_csv(os.path.join(self.raw_dir, self.test_raw))

        # Drop the non descriptive columns
        train = train.drop(["timestamp"], axis=1)

        # Renaming mispelled columns
        train = train.rename(columns={"reactice_power_calculated_by_converter": "reactive_power_calculated_by_converter"})
        test = test.rename(columns={"reactice_power_calculated_by_converter": "reactive_power_calculated_by_converter"})

        # Drop the rows with missing values
        # There are no missing values in this dataset

        # Split the train dataset into kfolds
        train = self._create_folds(train)

        # Save the cleaned dataset
        train.to_csv(os.path.join(self.clean_dir, self.train_clean), index=False)
        test.to_csv(os.path.join(self.clean_dir, self.test_clean), index=False)

    def _create_folds(self, data):
        """Create folds for cross-validation"""

        # create the new kfold column
        data["kfold"] = -1

        # randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True)

        # initiate the kfold class from model_selection module
        kf = model_selection.KFold(n_splits=self.fold_num)

        # fill the new kfold column
        for f, (t_, v_) in enumerate(kf.split(X=data)):
            data.loc[v_, "kfold"] = f

        return data

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config.yaml")
    parsed_args = args.parse_args()

    DataCleaner(config_path=parsed_args.config).clean_dataset()