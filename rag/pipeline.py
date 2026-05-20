from rag.prompts import build_rag_prompt
from rag.retriever import create_retriever, load_vector_store
from rag.models import create_answer_llm, create_judge_llm
from langchain_classic.schema.document import Document

DEFAULT_TOP_K = 3

    

def initialize_rag_pipeline(top_k: int = DEFAULT_TOP_K) -> dict:
    state = {}

   
    state['vector_store'] = load_vector_store()
    state['retriever'] = create_retriever(
        vector_store=state['vector_store'],
        top_k=top_k,
    )
    state['top_k'] = top_k
    state['answer_llm'] = create_answer_llm()
    state['judge_llm'] = create_judge_llm()

    return state 

def run_retrieval_only(question:str, pipeline_state:dict)->list[Document]:

   retriever = pipeline_state['retriever']

   return retriever.invoke(question)


def generate_answer(
        question:str,
        documents:list[Document],
        llm,
)->str:
    
    prompt = build_rag_prompt(question=question, documents=documents)

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content

    return str(response)

def extract_sources(documents:list[Document]) ->list[dict]:

    sources = []

    for document in documents:

        sources.append({
            "file_name": document.metadata.get("file_name", "unknown_source"),
            "source": document.metadata.get("source", "unknown_path"),
            "document_type":document.metadata.get("document_type", "unknown_type")
        }
        )
    
    return sources

def run_pipeline(question:str, pipeline_state:dict)->dict:

    relevant_documents = run_retrieval_only(
        question=question, 
        pipeline_state=pipeline_state,
    )

    answer = generate_answer(question=question, 
                documents=relevant_documents,
                llm=pipeline_state['answer_llm'])
    
    return {
        "question": question,
        "answer": answer,
        "sources": extract_sources(relevant_documents),
        "retrieved_documents": relevant_documents,
    }

if __name__ == "__main__":

    rag_state = initialize_rag_pipeline()

    result = run_pipeline(question="What is ColliderOpsAI and what problem does it solve?",
                        pipeline_state=rag_state,)

    print(result['answer'])
    print(result['sources'])
