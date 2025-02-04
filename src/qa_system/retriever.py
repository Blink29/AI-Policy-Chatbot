from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ..config.settings import (
    MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    PERSIST_DIR,
    RETRIEVAL_K
)

embeddings = OllamaEmbeddings(model=MODEL_NAME)
vector_store = Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR
)

def retrieve_docs(query, k=RETRIEVAL_K):
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        raise Exception(f"Error retrieving documents: {str(e)}")

def index_documents(documents):
    try:
        vector_store.add_documents(documents)
    except Exception as e:
        raise Exception(f"Error indexing documents: {str(e)}")
