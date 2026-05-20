from rag.build_vector_index import create_embeddings, PERSIST_DIRECTORY
from rag.config import RetrievalConfig
from langchain_classic.schema import BaseRetriever
from langchain_classic.vectorstores import Chroma


def load_vector_store():

    embedding_model = create_embeddings()

    vector_store = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embedding_model,
    )

    return vector_store

def create_retriever(vector_store:Chroma, top_k:int = RetrievalConfig.TOP_K)->BaseRetriever:

    retriever = vector_store.as_retriever(
        search_type=RetrievalConfig.STRATEGY,
        search_kwargs = {"k":top_k}
    )

    return retriever

def retrieve_documents(query:str, retriever):

    return retriever.invoke(input=query)



