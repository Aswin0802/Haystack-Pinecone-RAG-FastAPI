from haystack import Pipeline
from haystack.utils import Secret

from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder

from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever
)

from haystack_integrations.components.generators.huggingface_api import (
    HuggingFaceAPIChatGenerator
)

from haystack_integrations.common.huggingface_api.utils import (
    HFGenerationAPIType
)

from haystack.dataclasses import ChatMessage

from QASystem.utils import pinecone_config


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt_template = [
    ChatMessage.from_user(
        """
Answer the following question using ONLY the provided context.

If the context does not contain the answer, reply with:
"I don't know"

Question:
{{query}}

Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Answer:
"""
    )
]


# --------------------------------------------------
# RAG Function
# --------------------------------------------------

def get_result(query):

    query_pipeline = Pipeline()

    # 1. Query Embedder
    query_pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder()
    )

    # 2. Pinecone Retriever
    query_pipeline.add_component(
        "retriever",
        PineconeEmbeddingRetriever(
            document_store=pinecone_config(),
            top_k=5
        )
    )

    # 3. Chat Prompt Builder
    query_pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=prompt_template
        )
    )

    # 4. LLM
    query_pipeline.add_component(
        "llm",
        HuggingFaceAPIChatGenerator(
            api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,

            api_params={
                "model": "Qwen/Qwen3.5-9B",
                "provider": "together"
            },

            token=Secret.from_env_var("HF_TOKEN"),

            generation_kwargs={
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.8,
                "extra_body": {
                    "top_k": 20,
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            }
        )
    )

    # --------------------------------------------------
    # Connections
    # --------------------------------------------------

    query_pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    query_pipeline.connect(
        "retriever.documents",
        "prompt_builder.documents"
    )

    query_pipeline.connect(
        "prompt_builder.prompt",
        "llm.messages"
    )

    # --------------------------------------------------
    # Run pipeline
    # --------------------------------------------------

    results = query_pipeline.run(
        {
            "text_embedder": {
                "text": query
            },

            "prompt_builder": {
                "query": query
            }
        }
    )

    # --------------------------------------------------
    # Get only LLM answer
    # --------------------------------------------------

    message = results["llm"]["replies"][0]


    return message.text


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    result = get_result("what is RAG?")

    print("\nFINAL RESULT:")
    print(result)