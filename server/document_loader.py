import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "sample.pdf")
chroma_db_path = os.path.join(project_root, "chroma_db")

def get_vector_store():
    """Initialize and return the vector store (Chroma database)"""
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    return vectorstore

def initialize_documents():
    """Load documents and create vector store if not already created"""
    try:
        pdf_loader = PyPDFLoader(data_path)
        documents = pdf_loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        print("Attempting to use text file as fallback...")
        text_path = os.path.join(project_root, "data", "sample.txt")
        text_loader = TextLoader(text_path)
        documents = text_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    
    #vector store
    db = Chroma.from_documents(texts, embeddings, persist_directory=chroma_db_path)
    print("Document loaded and vector store created successfully.")
    return db

# Initialize vector store on import
db = get_vector_store()