import os, json
from uuid import uuid4

import boto3
import time
from aws_lambda_powertools import Logger
from langchain_aws import BedrockEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone

DOCUMENT_TABLE = os.environ["DOCS_TABLE"]
pinecone_api_key = os.environ['KNOWLEDGE_BASE_ID']
BUCKET = os.environ["BUCKET"]

s3 = boto3.client("s3")

ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
logger = Logger()


def set_doc_status(user_id, document_id, status):
    document_table.update_item(
        Key={"userId": user_id, "documentId": document_id},
        UpdateExpression="SET docstatus = :docstatus",
        ExpressionAttributeValues={":docstatus": status},
    )


index_name = 'rag-with-bedrock-pinecone'


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    event_body = json.loads(event["Records"][0]["body"])
    document_id = event_body["documentId"]

    user_id = event_body["user"]
    key = event_body["key"]
    file_name_full = key.split("/")[-1]

    set_doc_status(user_id, document_id, "PROCESSING")

    s3.download_file(BUCKET, key, f"/tmp/{file_name_full}")

    loader = PyPDFLoader(f"/tmp/{file_name_full}")

    documents = loader.load()

    pc = Pinecone(api_key=pinecone_api_key)
    spec = ServerlessSpec(cloud='aws', region='us-east-1')

    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        # create a new index
    pc.create_index(
        index_name,
        dimension=1536,
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1"
    )

    logger.info(f"loaded documents are... {documents}")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    docs = text_splitter.split_documents(documents)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    index = pc.Index(name=index_name)

    logger.info(f"embeddings... {documents}")
    vector_store = PineconeVectorStore(embedding=embeddings,
                                       index=index)
    vector_store.add_documents(
        documents=docs

    )

    logger.info(f"more documents are... {documents}")

    logger.info(f"Indexing {file_name_full}...")

    set_doc_status(user_id, document_id, "READY")