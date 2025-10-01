# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=str, required=True, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=str, required=True, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args = parser.parse_args()

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    mlflow.start_run()

    print(f"Registering model: {args.model_name}")
    print(f"Input model_path:  {args.model_path}")
    
    model = mlflow.sklearn.load_model(args.model_path)

    logged = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
    )
    model_uri = logged.model_uri

    registered = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    Path(args.model_info_output_path).mkdir(parents=True, exist_ok=True)
    info_path = Path(args.model_info_output_path) / "model_info.json"

    client = MlflowClient()
    model_version = registered.version
    model_name = registered.name
    model_details = {
        "model_name": model_name,
        "model_version": model_version,
        "model_uri": model_uri,
    }
    with open(info_path, "w") as f:
        json.dump(model_details, f)

    print(f"Registered {model_name} v{model_version}")
    print(f"Wrote: {info_path}")
    
    mlflow.end_run()

    # -----------  WRITE YOR CODE HERE -----------
    
    # Step 1: Load the model from the specified path using `mlflow.sklearn.load_model` for further processing.  
    # Step 2: Log the loaded model in MLflow with the specified model name for versioning and tracking.  
    # Step 3: Register the logged model using its URI and model name, and retrieve its registered version.  
    # Step 4: Write model registration details, including model name and version, into a JSON file in the specified output path.  


if __name__ == "__main__":
    
    args = parse_args()
    print(f"Model name:             {args.model_name}")
    print(f"Model path:             {args.model_path}")
    print(f"Model info output path: {args.model_info_output_path}")
    main(args)
