# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import os
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

def _pick_csv(path: str) -> str:
    files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV found in {path}. Contents: {os.listdir(path)}")
    return os.path.join(path, files[0])

def _find_target_name(cols, name="Price"):
    # Case-insensitive match for "Price"
    for c in cols:
        if c.lower() == name.lower():
            return c
    raise KeyError(f"Target column '{name}' not found. Columns: {list(cols)}")

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods.  
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.  
    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.  
    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.  
    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.  
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.  

    started_here = False
    if mlflow.active_run() is None:
        mlflow.start_run()
        started_here = True

    print("[train] active run id:", mlflow.active_run().info.run_id)
  
    # Read data (each input is a folder -> pick the CSV inside - defined in previous step)
    train_df = pd.read_csv(_pick_csv(args.train_data))
    test_df = pd.read_csv(_pick_csv(args.test_data))

    # Split features/target
    target_col = _find_target_name(train_df.columns, "Price")
    train_df[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
    test_df[target_col] = pd.to_numeric(test_df[target_col], errors="coerce")
    train_df = train_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Train
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    ).fit(X_train, y_train)

    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    print(f"The MSE of the random forest regressor on the test set is: {mse:.4f}")
    # IMPORTANT: The primary_metric is "MSE" 
    mlflow.log_metric("MSE", mse)

    Path(args.model_output).mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

    if started_here:
        mlflow.end_run()

if __name__ == "__main__":
    args = parse_args()
    print(f"Train dataset input path: {args.train_data}")
    print(f"Test dataset input path:  {args.test_data}")
    print(f"Model output path:        {args.model_output}")
    print(f"n_estimators:              {args.n_estimators}")
    print(f"max_depth:                 {args.max_depth}")
    main(args)
