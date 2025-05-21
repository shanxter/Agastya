import json
import os
import logging
import uuid # To generate unique IDs for chunks

# Using LangChain for text splitting
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # from langchain_text_splitters import RecursiveCharacterTextSplitter # Newer import path
except ImportError:
    logging.error("LangChain not installed. Please install it: pip install langchain")
    exit()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Input file from run_content_ingestion.py
# Assuming this script is also in Agastya/ingest_pipeline/
RAW_CONTENT_INPUT_JSONL = "all_indications_raw_content.jsonl"

# Output file for chunked data
CHUNKED_OUTPUT_JSONL = "chunked_content_for_embedding.jsonl"

# Chunking Parameters - TUNE THESE
CHUNK_SIZE = 1000  # Target characters per chunk
CHUNK_OVERLAP = 150 # Characters of overlap between chunks
# For token-based splitting with RecursiveCharacterTextSplitter or TokenTextSplitter,
# you'd use a tokenizer like tiktoken:
# import tiktoken
# tokenizer = tiktoken.get_encoding("cl100k_base")
# def tiktoken_len(text):
#    tokens = tokenizer.encode(text, disallowed_special=())
#    return len(tokens)
# CHUNK_SIZE = 256 # Target tokens per chunk
# CHUNK_OVERLAP = 30 # Target tokens of overlap

# --- Text Splitter Initialization ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,  # Use simple character length
    # length_function=tiktoken_len, # If using token-based length with above tokenizer
    add_start_index=True, # Adds 'start_index' to metadata of each chunk
    separators=["\n\n", "\n", ". ", " ", ""] # Default good separators
)

# --- Main Chunking Logic ---
def chunk_documents(input_filepath, output_filepath):
    """
    Reads raw documents from input_filepath, chunks them, and writes to output_filepath.
    """
    all_chunked_data = []
    total_docs_processed = 0
    total_chunks_created = 0

    input_path = os.path.join(os.path.dirname(__file__), input_filepath)
    output_path = os.path.join(os.path.dirname(__file__), output_filepath)

    logging.info(f"Starting chunking process. Input: {input_path}, Output: {output_path}")
    logging.info(f"Chunker settings: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, length_fn=len (chars)")

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line_number, line in enumerate(infile, 1):
                try:
                    original_doc = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line {line_number} in {input_path}: {line.strip()}")
                    continue

                doc_text = original_doc.get("text", "")
                doc_title = original_doc.get("title", "N/A")

                # Prepend title to the text to give chunks more context, if desired
                # This can be very effective.
                text_to_chunk = f"Title: {doc_title}\n\n{doc_text}" if doc_title != "N/A" else doc_text

                if not text_to_chunk.strip():
                    logging.warning(f"Skipping document with empty text (original ID: {original_doc.get('id', 'Unknown')}, line: {line_number}).")
                    continue
                
                # Use LangChain's text splitter
                # The splitter expects a string or a list of LangChain Document objects.
                # We'll split string by string.
                # For metadata handling with splitter.split_text, we pass it separately.
                
                chunks_texts = text_splitter.split_text(text_to_chunk)
                
                # Create standardized chunk objects
                for chunk_index, chunk_text_content in enumerate(chunks_texts):
                    chunk_id = str(uuid.uuid4()) # Generate a unique ID for each chunk

                    # Inherit metadata from the parent document
                    # Add chunk-specific metadata
                    chunk_metadata = {
                        "original_doc_id": original_doc.get("id", "N/A"),
                        "original_doc_source": original_doc.get("source", "N/A"),
                        "original_doc_title": doc_title,
                        "original_doc_url": original_doc.get("url", "N/A"),
                        "original_doc_publication_date": original_doc.get("publication_date"),
                        "original_doc_retrieved_at": original_doc.get("retrieved_at"),
                        "chunk_index_in_doc": chunk_index,
                        # 'start_index' might be added by text_splitter if add_start_index=True
                        # and the splitter.create_documents() or splitter.split_documents() method is used.
                        # Since we use split_text, we might not get it directly here unless we simulate Document creation.
                        # Let's keep it simple for now.
                    }
                    # If 'add_start_index' was True and we used split_documents, it would be in chunk.metadata['start_index']
                    # For split_text, we'd need to adapt or manually track if highly desired.
                    # For simplicity, we will not rely on `start_index` from `split_text` directly.

                    # Include any other "metadata" from the original document
                    if "metadata" in original_doc and isinstance(original_doc["metadata"], dict):
                         chunk_metadata.update(original_doc["metadata"])


                    chunk_data = {
                        "chunk_id": chunk_id,
                        "text_chunk": chunk_text_content.strip(),
                        "metadata": chunk_metadata
                    }
                    all_chunked_data.append(chunk_data)
                    total_chunks_created += 1
                
                total_docs_processed += 1
                if total_docs_processed % 100 == 0:
                    logging.info(f"Processed {total_docs_processed} documents, created {total_chunks_created} chunks so far...")

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        return
    except Exception as e:
        logging.error(f"An error occurred during chunking: {e}", exc_info=True)
        return

    logging.info(f"Finished processing {total_docs_processed} documents.")
    logging.info(f"Total chunks created: {total_chunks_created}.")

    # Write chunked data to output file
    if all_chunked_data:
        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for chunk_item in all_chunked_data:
                    outfile.write(json.dumps(chunk_item) + '\n')
            logging.info(f"Successfully wrote {len(all_chunked_data)} chunks to {output_path}")
        except IOError as e:
            logging.error(f"Error writing chunked data to {output_path}: {e}")
    else:
        logging.info("No chunks were created to write.")

if __name__ == "__main__":
    # Ensure the script knows its own directory for relative paths
    script_dir = os.path.dirname(__file__)
    if not script_dir: 
        script_dir = os.getcwd()
    
    input_full_path = os.path.join(script_dir, RAW_CONTENT_INPUT_JSONL)
    
    if not os.path.exists(input_full_path):
        logging.error(f"CRITICAL: Raw content input file '{RAW_CONTENT_INPUT_JSONL}' not found in the script directory: {script_dir}")
        logging.error("Please run 'run_content_ingestion.py' first to generate this file.")
    else:
        chunk_documents(RAW_CONTENT_INPUT_JSONL, CHUNKED_OUTPUT_JSONL)