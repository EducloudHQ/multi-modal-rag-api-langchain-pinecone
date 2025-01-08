import os, json
import shortuuid
import boto3
from pathlib import Path
from aws_lambda_powertools import Logger
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Local utility for Pinecone index creation
from utilities.pinecone_utils import create_or_recreate_index

DOCUMENT_TABLE = os.environ["DOCS_TABLE"]
bedrock = boto3.client("bedrock-runtime", "us-east-1")
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", "us-east-1")

BUCKET = os.environ["BUCKET"]

s3 = boto3.client("s3")

ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
logger = Logger()

index_name = 'rag-with-bedrock-pinecone'

def set_doc_status(user_id: str, document_id: str, status: str) -> None:
    """
    Update the docstatus field in DynamoDB for the specified user/document.
    """
    document_table.update_item(
        Key={"userId": user_id, "documentId": document_id},
        UpdateExpression="SET docstatus = :docstatus",
        ExpressionAttributeValues={":docstatus": status},
    )

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    logger.info(f"event is {event}")
    key = event["Key"]
    document_id = shortuuid.uuid()
    s3.download_file(BUCKET, key, f"/tmp/{key}")

    data = json.loads(Path(f"/tmp/{key}").read_text())

    transcript = data['results']['transcripts'][0]['transcript']

    logger.info(f"loaded data {data['results']['transcripts'][0]['transcript']}")
    set_doc_status("rosius", document_id, "PROCESSING")
    # Create or recreate the Pinecone index
    index = create_or_recreate_index(
        index_name=index_name,
        dimension=1536,  # Titan embedding dimension
        metric="dotproduct",
        region="us-east-1",
        cloud_provider="aws",
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1"
    )
    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([transcript])

    logger.info(f"Found {len(docs)}, {docs}")

    logger.info(f"Found {len(docs)}, {docs}")

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )
    vector_store.add_documents(documents=docs, async_req=False)
    set_doc_status("rosius", document_id, "COMPLETED")