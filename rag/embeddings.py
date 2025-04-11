from langchain.embeddings import HuggingFaceEmbeddings

# Initialize free embedding model
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # No GPU required
    encode_kwargs={'normalize_embeddings': False}
)
