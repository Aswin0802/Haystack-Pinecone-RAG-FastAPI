import os
from haystack.utils import Secret
from dotenv import load_dotenv
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the .env file")


def pinecone_config():
    """
    Create and return the Pinecone document store.
    """

    document_store = PineconeDocumentStore(
    index="haystack",
    namespace="rag",
    metric="cosine",
    dimension=768,
    spec={
        "serverless": {
            "region": "us-east-1",
            "cloud": "aws"
        }
    },
    api_key=Secret.from_env_var("PINECONE_API_KEY"),
    )

    return document_store