
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack.components.converters import PyPDFToDocument
from pathlib import Path # type: ignore
from QASystem.utils import pinecone_config
from dotenv import load_dotenv
load_dotenv()

def ingest(document_store):

	#configuring pinecone database
	'''document_store = PineconeDocumentStore(
		environment="gcp-starter",
		index="default",
		namespace="default",
		dimension=768
	)
	'''

	#creating a pipeline object
	indexing = Pipeline()

	#adding the components in pipeline
	indexing.add_component("converter", PyPDFToDocument())
	indexing.add_component("splitter", DocumentSplitter(split_by="word", split_length=200,split_overlap=30))
	indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
	indexing.add_component("writer", DocumentWriter(document_store))

	#coneecting all the components of pipeline
	indexing.connect("converter", "splitter")
	indexing.connect("splitter", "embedder")
	indexing.connect("embedder", "writer")

	#stroing the data as a embedding in the database
	indexing.run({"converter": {"sources": [Path("C:/Users/Aswin Kumar Nayak/OneDrive/Documents/AI Frameworks/Complete RAG/Simple_RAG/Haystack_Mistral_Pinecone_FastAPI/Data/RAG.pdf")]}})
 
 
if __name__ == "__main__":
    #loading the environment variable
    '''load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    
    print("Import Successfully")'''
    document_store=pinecone_config()
    
    ingest(document_store)