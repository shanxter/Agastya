import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv # To load OPENAI_API_KEY for the embedding function

# Import model configuration (adjust the import path based on your structure)
try:
    from agent.model_config import get_model_for_task
    logging.info("Successfully imported model configuration")
except ImportError:
    logging.error("Could not import model configuration, using default embedding model")
    # Define a fallback function if import fails
    def get_model_for_task(task):
        return "text-embedding-3-small" if task == "embeddings" else "gpt-3.5-turbo"

# --- .env loading ---
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
# --- END .env loading ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Configuration (could eventually move to a config file)
# Path relative to this file (tools/research_tool.py) to db/chroma_store
CHROMA_DB_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db", "chroma_store")
CHROMA_COLLECTION_NAME = "hcp_knowledge_base"
OPENAI_EMBEDDING_MODEL = get_model_for_task("embeddings")  # Use configured model instead of hardcoded value
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not found. Research tool cannot initialize embeddings.")
    # Decide how to handle this - raise error or return empty?
    # For now, it will cause an error when OpenAIEmbeddings is initialized.

try:
    logging.info(f"Initializing embeddings with model: {OPENAI_EMBEDDING_MODEL}")
    embedding_function = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_PERSIST_DIRECTORY
    )
    # Create a retriever from the vector store
    # k=5 means it will retrieve the top 5 most similar documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 30}) 
    logging.info(f"ChromaDB retriever initialized for collection '{CHROMA_COLLECTION_NAME}'.")
except Exception as e:
    logging.error(f"Failed to initialize ChromaDB retriever: {e}", exc_info=True)
    retriever = None # Set to None if initialization fails

def query_chroma_db(research_query: str) -> list[dict]:
    """
    Queries the ChromaDB for documents relevant to the research_query.
    Returns a list of dictionaries, where each dict contains 'page_content' and 'metadata'.
    """
    if not retriever:
        logging.error("ChromaDB retriever is not available. Cannot perform search.")
        return []
    if not research_query or not isinstance(research_query, str):
        logging.warning("Invalid research query provided.")
        return []

    try:
        logging.info(f"Research tool: Searching ChromaDB for: '{research_query}'")
        # .invoke() is the standard way to call retrievers in newer LangChain
        retrieved_langchain_docs = retriever.invoke(research_query) 
        
        results = []
        for doc in retrieved_langchain_docs:
            results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata 
            })
        logging.info(f"Research tool: Retrieved {len(results)} documents.")
        return results
    except Exception as e:
        logging.error(f"Error during ChromaDB query: {e}", exc_info=True)
        return []

# Example Test (optional, for direct testing of this file)
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Please set your OPENAI_API_KEY environment variable to test.")
    else:
        test_query = "latest treatments for lung cancer"
        print(f"Testing research_tool with query: '{test_query}'")
        print(f"Using embedding model: {OPENAI_EMBEDDING_MODEL}")
        docs = query_chroma_db(test_query)
        if docs:
            for i, doc_data in enumerate(docs):
                print(f"\n--- Document {i+1} ---")
                print(f"Content (Snippet): {doc_data['page_content'][:250]}...")
                print(f"Metadata: {doc_data['metadata']}")
        else:
            print("No documents retrieved or an error occurred.")