import json
import os
import logging
import time
import uuid # For fallback ID generation if needed

# --- START .env loading ---
from dotenv import load_dotenv 

# Calculate the path to the .env file in the parent directory (Agastya/)
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    # print(f"DEBUG (embed_and_store.py): Loaded environment variables from: {dotenv_path}") # Uncomment for debug
else:
    # print(f"DEBUG (embed_and_store.py): .env file not found at {dotenv_path}, relying on system environment variables.") # Uncomment for debug
    pass
# --- END .env loading ---


# LangChain components for embeddings and vector store
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document 
    from langchain_community.vectorstores.utils import filter_complex_metadata # For cleaning metadata
except ImportError:
    # Configure basic logging if critical error occurs before full config
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO)
    logging.critical("LangChain (openai, community, core, utils) not installed. Please ensure all are installed: pip install langchain-openai langchain-community langchain-core chromadb python-dotenv")
    exit()

# ChromaDB client (though LangChain wrapper often handles direct interaction)
try:
    import chromadb
except ImportError:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO)
    logging.critical("ChromaDB not installed. Please install it: pip install chromadb")
    exit()

# --- Configuration ---
# Ensure logging is configured AFTER potential .env loading but before first use if not already set by imports
if not logging.getLogger().hasHandlers(): 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Input file from chunk_text.py
CHUNKED_INPUT_JSONL = "chunked_content_for_embedding.jsonl"

# ChromaDB Configuration
CHROMA_DB_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db", "chroma_store")
CHROMA_COLLECTION_NAME = "hcp_knowledge_base" 

# OpenAI Embedding Model
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Batch size for adding documents to ChromaDB
BATCH_SIZE_CHROMA_ADD = 100 

# --- OpenAI API Key Check (This now comes AFTER .env loading) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.critical("OPENAI_API_KEY environment variable not set (or not loaded from .env). This script cannot run.")
    exit()

# --- Initialize Embedding Function ---
embedding_function = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY
    # dimensions=512 # Optional: For text-embedding-3-small, can reduce dimensions if needed
)

# --- Main Embedding and Storing Logic ---
def embed_and_store_chunks(chunked_input_filepath, persist_directory, collection_name):
    """
    Reads chunked documents, generates embeddings, and stores them in ChromaDB.
    """
    try:
        os.makedirs(persist_directory, exist_ok=True)
        logging.info(f"ChromaDB persistence directory ensured at: {persist_directory}")
    except OSError as e:
        logging.error(f"Could not create ChromaDB persistence directory {persist_directory}: {e}")
        return

    # Use absolute path for input file for robustness
    input_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), chunked_input_filepath))
    logging.info(f"Starting embedding and storing process. Input: {input_path}")
    logging.info(f"ChromaDB collection: '{collection_name}', Persist dir: '{persist_directory}'")
    logging.info(f"Using OpenAI embedding model: '{OPENAI_EMBEDDING_MODEL}'")

    documents_to_embed = []
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_number, line in enumerate(infile, 1):
                try:
                    chunk_data = json.loads(line)
                    
                    raw_metadata = chunk_data.get("metadata", {})
                    if not isinstance(raw_metadata, dict):
                        logging.warning(f"Metadata for chunk at line {line_number} is not a dict. Got: {type(raw_metadata)}. Replacing with empty dict.")
                        raw_metadata = {}
                    
                    # Create an initial Document object with potentially complex metadata
                    doc_to_filter = Document(
                        page_content=chunk_data.get("text_chunk", ""),
                        metadata=raw_metadata.copy() # Pass a copy to avoid modifying original dict if needed elsewhere
                    )

                    # Apply LangChain's utility to filter complex metadata.
                    # For a single Document input, it returns a single Document output.
                    filtered_doc = filter_complex_metadata([doc_to_filter])[0]
                                        
                    # Ensure 'chunk_id' from the original chunk_data is in the filtered_doc.metadata.
                    # filter_complex_metadata should preserve simple string values like chunk_id.
                    # This step re-adds it if it was somehow removed or to ensure it's top-level.
                    original_chunk_id = chunk_data.get("chunk_id")
                    if original_chunk_id:
                        filtered_doc.metadata["chunk_id"] = str(original_chunk_id) # Ensure it's a string
                    else:
                        # This case should ideally not happen if chunk_text.py always assigns a chunk_id
                        logging.warning(f"Original chunk_data at line {line_number} missing 'chunk_id'. Generating a new one for Document metadata.")
                        filtered_doc.metadata["chunk_id"] = str(uuid.uuid4())


                    if not filtered_doc.page_content.strip():
                        logging.warning(f"Skipping empty text_chunk at line {line_number} from original doc ID: {filtered_doc.metadata.get('original_doc_id', 'Unknown')}")
                        continue
                    
                    documents_to_embed.append(filtered_doc)

                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line {line_number} in {input_path}: {line.strip()}")
                    continue
                except Exception as e:
                    logging.error(f"Error processing line {line_number} from {input_path}: {e}", exc_info=True)
                    continue # Skip this problematic chunk and try to continue with others
    except FileNotFoundError:
        logging.error(f"Chunked input file not found: {input_path}")
        return
    
    if not documents_to_embed:
        logging.info("No documents found to embed after processing input file. Exiting.")
        return

    logging.info(f"Loaded {len(documents_to_embed)} chunks to be embedded and stored.")
    
    logging.info(f"Initializing Chroma vector store...")
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
        logging.info(f"Chroma vector store initialized/loaded. Collection items before add: {vector_store._collection.count()}")

        total_docs = len(documents_to_embed)
        for i in range(0, total_docs, BATCH_SIZE_CHROMA_ADD):
            batch = documents_to_embed[i:i + BATCH_SIZE_CHROMA_ADD]
            
            batch_ids = []
            for doc_idx, doc_in_batch in enumerate(batch):
                # Use the chunk_id we ensured is in the metadata
                chunk_id_val = doc_in_batch.metadata.get("chunk_id")
                if chunk_id_val:
                    batch_ids.append(str(chunk_id_val)) # Ensure it's string
                else:
                    # This is a fallback, indicates an issue if chunk_id wasn't set properly
                    fallback_id = str(uuid.uuid4()) # Generate a truly unique ID
                    batch_ids.append(fallback_id)
                    logging.error(f"CRITICAL: 'chunk_id' was unexpectedly missing from Document metadata for content: '{doc_in_batch.page_content[:50]}...'. Using generated fallback ID: {fallback_id}")

            logging.info(f"Adding batch {i // BATCH_SIZE_CHROMA_ADD + 1}/{(total_docs + BATCH_SIZE_CHROMA_ADD -1) // BATCH_SIZE_CHROMA_ADD} to Chroma ({len(batch)} documents)...")
            
            vector_store.add_documents(documents=batch, ids=batch_ids)
            
            logging.info("Persisting ChromaDB batch...")
            vector_store.persist() 
            logging.info("ChromaDB batch persisted.")
            # time.sleep(0.5) # Optional small delay if hitting rate limits (unlikely for OpenAI embeddings here)

        logging.info(f"All documents processed. Final collection count: {vector_store._collection.count()}")
        logging.info("Embedding and storing process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during ChromaDB initialization or document addition: {e}", exc_info=True)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    
    chunked_file_full_path = os.path.join(script_dir, CHUNKED_INPUT_JSONL)
    # Ensure CHROMA_DB_PERSIST_DIRECTORY is resolved to an absolute path for consistency
    chroma_db_full_persist_dir = os.path.abspath(CHROMA_DB_PERSIST_DIRECTORY) 

    if not os.path.exists(chunked_file_full_path):
        logging.error(f"CRITICAL: Chunked input file '{CHUNKED_INPUT_JSONL}' not found at: {chunked_file_full_path}")
        logging.error("Please run 'chunk_text.py' first to generate this file.")
    else:
        logging.info(f"Using ChromaDB persistence directory: {chroma_db_full_persist_dir}")
        embed_and_store_chunks(
            chunked_input_filepath=CHUNKED_INPUT_JSONL, # This will be joined with script_dir inside the function
            persist_directory=chroma_db_full_persist_dir, # Pass absolute path
            collection_name=CHROMA_COLLECTION_NAME
        )