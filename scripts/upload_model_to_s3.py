import boto3
import numpy as np
import pandas as pd
s3 = boto3.client("s3")

bucket_name = "pycatanbucket"


model_path = "working_model.keras"
s3.delete_object(Bucket=bucket_name, Key=f"models/working_model.keras") 
try:
    # Head object to check if the file exists
    s3.head_object(Bucket=bucket_name, Key=f"models/working_model.keras")
    print(f"File 'models/working_model.keras' exists in bucket '{bucket_name}'")
except:
    print(f"File 'models/working_model.keras' does not exist in bucket '{bucket_name}'")
s3.upload_file(model_path, bucket_name, f"models/working_model.keras")
try:
    # Head object to check if the file exists
    s3.head_object(Bucket=bucket_name, Key=f"models/working_model.keras")
    print(f"File 'models/working_model.keras' exists in bucket '{bucket_name}'")
except:
    print(f"File 'models/working_model.keras' does not exist in bucket '{bucket_name}'")