import boto3
import os

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_KEY_ID']
)

for bucket in s3.buckets.all():
    print(bucket.name)
