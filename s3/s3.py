import boto3
import os

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_KEY_ID']
)

s3_resource = boto3.resource('s3')
bucket = s3.Bucket('pkxd-gsn')
for obj in bucket.objects.filter(Prefix='posts/images'):
    if not os.path.exists(os.path.dirname(obj.key)):
        os.makedirs(os.path.dirname(obj.key))
    bucket.download_file(obj.key, obj.key)
