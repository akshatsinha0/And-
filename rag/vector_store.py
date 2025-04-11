from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.embeddings import embed_model
from langchain.vectorstores import Chroma

# Load and split documents
loader = TextLoader("data/chat_history.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# Create vector store
vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embed_model,
    persist_directory="embeddings"
)
