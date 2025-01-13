import os
import json

import boto3
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.data_classes.appsync import scalar_types_utils
from aws_lambda_powertools.utilities.data_classes import event_source, SQSEvent

# LangChain & Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings

# Local utility for Pinecone index creation
from utilities.pinecone_utils import create_or_recreate_index

logger = Logger()

# Environment variables
DOCUMENT_TABLE = os.environ["DOCS_TABLE"]
BUCKET = os.environ["BUCKET"]
STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN")
INDEX_NAME = "rag-with-bedrock-pinecone"

# AWS Clients/Resources
s3 = boto3.client("s3")
step_function_client = boto3.client("stepfunctions")
ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)


def set_doc_status(user_id: str, document_id: str, status: str) -> None:
    """
    Update the docstatus field in DynamoDB for the specified user/document.
    """
    document_table.update_item(
        Key={"userId": user_id, "documentId": document_id},
        UpdateExpression="SET documentStatus = :docstatus",
        ExpressionAttributeValues={":docstatus": status},
    )


@event_source(data_class=SQSEvent)
@logger.inject_lambda_context(log_event=True)
def lambda_handler(event: SQSEvent, context):
    """
    Processes SQS events containing information about media or document files.
    If the file is .mp3 or .mp4, it triggers a Step Function workflow.
    Otherwise, it downloads the PDF from S3, creates (or recreates) a Pinecone
    index, splits the PDF into chunks, and inserts them into the vector store.
    """

    for record in event.records:
        logger.info(f"Received event: {record}")
        event_body = json.loads(record.body)
        logger.info(f"Received event body: {event_body}")

        extension = event_body.get("extension")
        s3_uri = event_body.get("s3_uri")
        #user_id = event_body.get("user")
        user_id = "UPLOADER_ID"
        document_id = event_body.get("documentId")
        key = event_body.get("key")

        # If file extension is .mp3 or .mp4, start a Step Functions workflow
        if extension in [".mp3", ".mp4"]:
            logger.info(f"Media file detected: {extension}, starting Step Function.")
            input_json = {
                "jobName": scalar_types_utils.make_id(),
                "bucketOutputKey": f"{scalar_types_utils.make_id()}.json",
                "mediaFileUri": s3_uri,
            }

            # Invoke step functions workflow
            step_function_client.start_execution(
                stateMachineArn=STATE_MACHINE_ARN,
                name=scalar_types_utils.make_id(),
                input=json.dumps(input_json),
            )
            continue

        # Otherwise, treat as PDF, process and index in Pinecone
        logger.info(f"Processing PDF document: {key}")

        # Update doc status to "PROCESSING"
        set_doc_status(user_id, document_id, "PROCESSING")

        # Download file from S3 into Lambda's /tmp/ directory
        file_name_full = key.split("/")[-1]  # e.g., "doc.pdf"
        local_path = f"/tmp/{file_name_full}"
        s3.download_file(BUCKET, key, local_path)

        # Load and split PDF
        loader = PyPDFLoader(local_path)
        documents = loader.load()
        logger.info(f"Loaded documents from {file_name_full}, total pages: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(docs)} document chunks.")

        # Create or recreate the Pinecone index
        index = create_or_recreate_index(
            index_name=INDEX_NAME,
            dimension=1536,  # Titan embedding dimension
            metric="dotproduct",
            region="us-east-1",
            cloud_provider="aws",
        )
        set_doc_status(user_id, document_id, "EMBEDDING")

        # Use LangChain's Bedrock embeddings
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

        # Build vector store
        logger.info(f"Building PineconeVectorStore for {INDEX_NAME}...")
        vector_store = PineconeVectorStore(embedding=embeddings, index=index)
        vector_store.add_documents(documents=docs, async_req=False)
        logger.info("Vector store updated with PDF contents.")

        # Update doc status to "READY"
        set_doc_status(user_id, document_id, "COMPLETED")
        logger.info(f"Document {document_id} for user {user_id} is READY.")
