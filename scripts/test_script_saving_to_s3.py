import boto3
import numpy as np
import pandas as pd
s3 = boto3.client("s3")

def save_to_s3(local_path, bucket, s3_path):
    s3.upload_file(local_path, bucket, s3_path)
    print(f"✅ Uploaded {local_path} to s3://{bucket}/{s3_path}")
bucket_name = "pycatanbucket"

data = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
csv_path = "data.csv"
df.to_csv(csv_path, index=False)

save_to_s3(csv_path, bucket_name, f"data/{csv_path}")

s3.download_file(bucket_name, f"data/{csv_path}", "downloaded_data.csv")
df = pd.read_csv("downloaded_data.csv")
print(df)