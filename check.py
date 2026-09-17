from dotenv import load_dotenv
load_dotenv()

from haystack import Document
from haystack.utils import Secret
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


document_store = PineconeDocumentStore(
    index="haystack",
    namespace="default",
    api_key=Secret.from_env_var("PINECONE_API_KEY"),
    dimension=768,
    metric="cosine",
    spec={
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    }
)

document = Document(
    content="This is a test document for Haystack and Pinecone.",
    embedding=[0.1] * 768
)

print("Before:", document_store.count_documents())

result = document_store.write_documents(
    [document],
    policy=DuplicatePolicy.OVERWRITE
)

print("write_documents result:", result)
print("After:", document_store.count_documents())