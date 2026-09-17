from haystack import Pipeline

from haystack.components.converters import (
    PyPDFToDocument
)

from haystack.components.preprocessors import (
    DocumentSplitter
)

from haystack.components.writers import (
    DocumentWriter
)

from haystack.document_stores.types import (
    DuplicatePolicy
)

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder
)

from utils import pinecone_config


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = (
    "C:/Users/Aswin Kumar Nayak/"
    "OneDrive/Documents/AI Frameworks/Complete RAG/"
    "Simple_RAG/Haystack_Mistral_Pinecone_FastAPI/"
    "Data/RAG.pdf"
)


# ============================================================
# INGEST FUNCTION
# ============================================================

def ingest(document_store):

    # --------------------------------------------------------
    # Create Pipeline
    # --------------------------------------------------------

    indexing = Pipeline()


    # --------------------------------------------------------
    # PDF → Documents
    # --------------------------------------------------------

    indexing.add_component(
        "converter",
        PyPDFToDocument()
    )


    # --------------------------------------------------------
    # Documents → Chunks
    # --------------------------------------------------------

    indexing.add_component(
        "splitter",
        DocumentSplitter(split_by="word",split_length=200,split_overlap=30)
    )


    # --------------------------------------------------------
    # Chunks → Embeddings
    # --------------------------------------------------------

    indexing.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder()
    )


    # --------------------------------------------------------
    # Embeddings → Pinecone
    # --------------------------------------------------------

    indexing.add_component(
        "writer",
        DocumentWriter(
            document_store=document_store,
            policy=DuplicatePolicy.OVERWRITE
        )
    )


    # --------------------------------------------------------
    # Connect Components
    # --------------------------------------------------------

    indexing.connect(
        "converter",
        "splitter"
    )

    indexing.connect(
        "splitter",
        "embedder"
    )

    indexing.connect(
        "embedder",
        "writer"
    )


    # --------------------------------------------------------
    # Run Pipeline
    # --------------------------------------------------------

    print("\nStarting ingestion...")

    result = indexing.run(
        {
            "converter": {
                "sources": [
                    PDF_PATH
                ]
            }
        }
    )


    # --------------------------------------------------------
    # Print Result
    # --------------------------------------------------------

    print("\nPipeline result:")
    print(result)

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nConnecting to Pinecone...")

    document_store = pinecone_config()


    print(
        "Documents before ingestion:",
        document_store.count_documents()
    )


    ingest(document_store)


    print(
        "\nDocuments after ingestion:",
        document_store.count_documents()
    )


    print("\nIngestion completed.")