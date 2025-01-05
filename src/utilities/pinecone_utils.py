import json
import time
import boto3
from pinecone import ServerlessSpec, Pinecone


def get_pinecone_api_key() -> str:
    """
    Fetch Pinecone API key from AWS Secrets Manager (example).
    Adjust the SecretId and region name based on your setup.
    """
    secret_name = "dev/pinecone-secret"  # Replace with your actual secret name
    region_name = "us-east-1"  # Replace with your secrets region

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    response = client.get_secret_value(SecretId=secret_name)
    secret_string = response["SecretString"]  # e.g. '{"PINECONE_API_KEY": "abc123..."}'
    secret_dict = json.loads(secret_string)

    # Adjust the key used here to match your secret's JSON structure
    return secret_dict.get("PINECONE_API_KEY", "")


def create_or_recreate_index(
        index_name: str,
        dimension: int = 1536,
        metric: str = "dotproduct",
        region: str = "us-east-1",
        cloud_provider: str = "aws",
) -> "pinecone.index.Index":
    """
    Creates a new Pinecone index, deleting any existing one with the same name.
    Waits until the index is ready before returning.

    :param index_name: Name of the index to create/recreate
    :param dimension: Dimensionality of the embeddings
    :param metric: Similarity metric, e.g. 'dotproduct', 'cosine', etc.
    :param region: Region for the ServerlessSpec
    :param cloud_provider: Cloud provider for the ServerlessSpec
    :return: A Pinecone Index object that is ready to use
    """
    api_key = get_pinecone_api_key()

    pc = Pinecone(api_key=api_key)
    spec = ServerlessSpec(cloud=cloud_provider, region=region)

    # Delete existing index if it exists
    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        pc.delete_index(index_name)

    # Create a new index
    pc.create_index(
        index_name,
        dimension=dimension,
        metric=metric,
        spec=spec
    )

    # Wait for index to become ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

    # Return a handle to the initialized index
    return pc.Index(name=index_name)
