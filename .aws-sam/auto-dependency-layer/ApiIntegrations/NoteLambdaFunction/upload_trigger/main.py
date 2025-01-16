import os, json
from datetime import datetime
import boto3
from pypdf import PdfReader
import shortuuid
from urllib.parse import unquote_plus
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.data_classes import event_source, S3Event
from aws_lambda_powertools.utilities.data_classes.appsync import scalar_types_utils

DOCUMENT_TABLE = os.environ["DOCS_TABLE"]
QUEUE = os.environ["QUEUE"]
BUCKET = os.environ["BUCKET"]

ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)

sqs = boto3.client("sqs")
s3 = boto3.client("s3")
logger = Logger()


@event_source(data_class=S3Event)
@logger.inject_lambda_context(log_event=True)
def lambda_handler(event: S3Event, context):
    logger.info(f"Received s3 event ${event}")
    bucket_name = event.bucket_name
    for record in event.records:
        logger.info(f"record is ${record}")

        key = unquote_plus(record.s3.get_object.key)

        root, extension = os.path.splitext(key)

        # Generate the S3 URI
        s3_uri = f"s3://{bucket_name}/{key}"

        logger.info(f"s3_uri is ${s3_uri}")

        document_id = shortuuid.uuid()
        timestamp = scalar_types_utils.aws_timestamp()

        logger.info(f"key is ${key}")

        document = {
            "id": document_id,
            "documentId": document_id,
            "documentName": key,
            "createdOn": timestamp,
            "documentType": extension,
            "userId":"UPLOADER_ID",

            "documentSize": record.s3.get_object.size,

            "documentStatus": "UPLOADED",

        }


        logger.info(document)

        document_table.put_item(Item=document)

        #stream -> Eventbridge Pipe -> EventRule -> Appsync(subscription)

        message = {
            "documentId": document_id,
            "key": key,
            "extension": extension,
            "root": root,
            "s3_uri": s3_uri,

        }
        sqs.send_message(QueueUrl=QUEUE, MessageBody=json.dumps(message))
