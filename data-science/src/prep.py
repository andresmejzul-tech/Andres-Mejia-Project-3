# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, required=True, help="Path to raw data")  # Specify the type for raw data (str)
    parser.add_argument("--train_data", type=str, required=True, help="Path to train dataset")  # Specify the type for train data (str)
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset")  # Specify the type for test data (str)
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")  # Specify the type (float) and default value (0.2) for test-train ratio
    args = parser.parse_args()

    return args

def encode_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object (string) columns, across train/test by fit on full data"""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    target_col = next((c for c in df.columns if c.lower() == "price"), None)
    if target_col is None:
        raise KeyError(f"'Price' column not found. Columns: {list(df.columns)}")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    df=encode_objects(df)
    
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

    print(f"Wrote: {train_path} and {test_path}")
    print("Dtypes after encoding:\n", train_df.dtypes)

    # ------- WRITE YOUR CODE HERE -------

    # Step 1: Perform label encoding to convert categorical features into numerical values for model compatibility.  
    # Step 2: Split the dataset into training and testing sets using train_test_split with specified test size and random state.  
    # Step 3: Save the training and testing datasets as CSV files in separate directories for easier access and organization.  
    # Step 4: Log the number of rows in the training and testing datasets as metrics for tracking and evaluation.  


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()  # Call the function to parse arguments

    lines = [
        f"Raw data path: {args.raw_data}",  # Print the raw_data path
        f"Train dataset output path: {args.train_data}",  # Print the train_data path
        f"Test dataset path: {args.test_data}",  # Print the test_data path
        f"Test-train ratio: {args.test_train_ratio}",  # Print the test_train_ratio
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
