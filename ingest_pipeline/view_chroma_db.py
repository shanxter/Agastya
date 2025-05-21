import os
import json
import sys
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"No .env file found at {dotenv_path}")

# Path to ChromaDB
CHROMA_DB_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db", "chroma_store")
CHROMA_COLLECTION_NAME = "hcp_knowledge_base"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

print(f"ChromaDB directory: {CHROMA_DB_PERSIST_DIRECTORY}")
print(f"Checking if directory exists: {os.path.exists(CHROMA_DB_PERSIST_DIRECTORY)}")

# List directory contents if it exists
if os.path.exists(CHROMA_DB_PERSIST_DIRECTORY):
    print("\nContents of ChromaDB directory:")
    for root, dirs, files in os.walk(CHROMA_DB_PERSIST_DIRECTORY):
        level = root.replace(CHROMA_DB_PERSIST_DIRECTORY, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

# Try to load the ChromaDB
try:
    print("\nTrying to load ChromaDB...")
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # Initialize the embedding function
    embedding_function = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Connect to the existing ChromaDB instance
    db = Chroma(
        persist_directory=CHROMA_DB_PERSIST_DIRECTORY,
        embedding_function=embedding_function,
        collection_name=CHROMA_COLLECTION_NAME
    )
    
    # Get the number of documents in the collection
    try:
        collection_count = db._collection.count()
        print(f"Total documents in collection: {collection_count}")
        
        # Get a sample document (the first one)
        if collection_count > 0:
            # Query the first document
            print("Running similarity search...")
            results = db.similarity_search("", k=1)
            
            if results:
                document = results[0]
                print("\n====== DOCUMENT CONTENT ======")
                print(f"Text: {document.page_content[:300]}...")
                print("\n====== METADATA ======")
                for key, value in document.metadata.items():
                    print(f"{key}: {value}")
            else:
                print("No results found in the query.")
        else:
            print("No documents found in the collection. Check if the embedding process has completed.")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    import traceback
    traceback.print_exc() 