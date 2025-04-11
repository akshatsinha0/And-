from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from rag.vector_store import vector_store

# Free model from Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    model_kwargs={"temperature":0.7, "max_new_tokens":500}
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

def get_response(query):
    result = qa_chain({"query": query})
    return result["result"]
