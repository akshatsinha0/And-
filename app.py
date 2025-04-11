from flask import Flask, request, jsonify, render_template
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Step 1: Prepare RAG components -------------------------------------------------
# Initialize embedding model
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Load and process documents
def initialize_rag():
    # Load your chat history/data
    loader = TextLoader("data/chat_history.txt")
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embed_model,
        persist_directory="embeddings"
    )
    
    # Initialize LLM from Hugging Face Hub (free tier)
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature":0.7, "max_new_tokens":500}
    )
    
    # Create RAG chain
    global qa_chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# Initialize RAG pipeline on app start
initialize_rag()

# Step 2: Flask routes -----------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Empty message received"}), 400
        
        # Get RAG response
        result = qa_chain({"query": user_message})
        ai_response = result["result"]
        
        return jsonify({"response": ai_response})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
