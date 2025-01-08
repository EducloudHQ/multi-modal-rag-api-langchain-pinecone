import json

import boto3
from aws_lambda_powertools import Logger
from aws_lambda_powertools.event_handler.appsync import Router
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrock
from langchain.llms.bedrock import Bedrock

from langchain.embeddings import BedrockEmbeddings
# Local utility for Pinecone index creation
from utilities.pinecone_utils import create_or_recreate_index
from langchain.chains import (
    RetrievalQA
)
from langchain_pinecone import PineconeVectorStore


logger = Logger()

logger = Logger(child=True)
router = Router()
bedrock = boto3.client("bedrock-runtime", "us-east-1")
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", "us-east-1")
INDEX_NAME = "rag-with-bedrock-pinecone"

@router.resolver(type_name="Query", field_name="queryDocument")
def query_document(input: str):
    # Setting Model kwargs
    model_kwargs = {
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

    llm = ChatBedrock(
        client=bedrock,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs=model_kwargs,
    )
    index = create_or_recreate_index(
        index_name=INDEX_NAME,
        dimension=1536,  # Titan embedding dimension
        metric="dotproduct",
        region="us-east-1",
        cloud_provider="aws",
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1"
    )
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    # Invoke Claude using the Langchain llm method


    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever(),
                                           )

    res = qa_chain({"query": input})

    logger.info(f"result is ${res}")

    return {"result": res['result'].replace("\n", "")}
