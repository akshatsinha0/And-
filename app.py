from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chardet
import os
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

def detect_file_encoding(file_path):
    """Detect file encoding using chardet"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))
    return result['encoding']

def format_zephyr_prompt(query, context):
    """Format prompt specifically for Zephyr models"""
    formatted_prompt = f"""<|system|>
You are a helpful AI assistant that accurately answers user questions based on the given context.
Use only the information from the context to answer, and say "I don't know" if the context doesn't contain the answer.
Context: {context}</s>
<|user|>
{query}</s>
<|assistant|>
"""
    return formatted_prompt

def initialize_rag():
    """Initialize RAG pipeline with large file handling"""
    try:
        file_path = os.path.abspath("data/chat_history.txt")
        print(f"Loading chat history from: {file_path}")
        
        encoding = detect_file_encoding(file_path)
        print(f"Detected encoding: {encoding}")

        loader = TextLoader(
            file_path=file_path,
            encoding=encoding,
            autodetect_encoding=True
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} initial documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        splits = []
        batch_size = 1000
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing chunks"):
            batch = documents[i:i+batch_size]
            splits.extend(text_splitter.split_documents(batch))
        
        print(f"Created {len(splits)} text chunks")

        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embed_model,
            persist_directory="embeddings"
        )
        print("Vector store initialized")

        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",  # Explicitly set the task
            max_length=1024,
            max_new_tokens=500,
            temperature=0.7,
            huggingfacehub_api_token=os.getenv("HF_TOKEN")
        )

        global qa_chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=format_zephyr_prompt("{question}", "{context}"),
                    input_variables=["context", "question"]
                )
            }
        )
        print("RAG pipeline ready")

    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise

initialize_rag()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "Empty message received"}), 400
        
        try:
            # Use invoke method instead of __call__
            result = qa_chain.invoke({"query": user_message})
            response = result["result"]
            cleaned_response = response.replace("<|im_end|>", "").strip()
            return jsonify({"response": cleaned_response})
        except Exception as chain_error:
            print(f"RAG Chain Error: {type(chain_error).__name__}: {str(chain_error)}")
            return jsonify({"error": "Unable to process your request with RAG"}), 500
    
    except Exception as e:
        error_type = type(e).__name__
        print(f"Chat error: {error_type}: {str(e)}")
        return jsonify({"error": f"AI service unavailable: {error_type}"}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
