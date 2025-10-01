# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, required=True, help="Folder produced by prep step (uri_folder)")
    parser.add_argument("--test_data", type=str, required=True, help="Folder produced by prep step (uri_folder)")
    parser.add_argument("--model_output", type=str, required=True, help="Output folder for the saved MLflow model")
    parser.add_argument("--n_estimators", type=int, default=100, help="RandomForest n_estimators")
    parser.add_argument("--max_depth", type=int, default=None, help="RandomForest max_depth (None for unlimited)")
     
    # -------- WRITE YOUR CODE HERE --------
    
    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters. Specify their types and defaults.  

    args = parser.parse_args()

    return args

def encode_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object (string) columns, across train/test """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods.  
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.  
    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.  
    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.  
    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.  
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.  

    # Reading Data
    df = pd.read_csv(args.raw_data)
    
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_train_ratio,
        random_state=42,
        shuffle=True,
        stratify=None 
    )
    
    train_dir = Path(args.train_data)
    test_dir = Path(args.test_data)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = train_dir / "data.csv"
    test_path  = test_dir / "data.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    mlflow.log_metric("train_rows", int(len(train_df)))
    mlflow.log_metric("test_rows", int(len(test_df)))
    
    with open(train_dir / "_columns.txt", "w") as f:
        f.write("\n".join(train_df.columns))

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
