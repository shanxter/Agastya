import json
import time
import os
import logging

# Assuming data_collector.py is in the same directory (Agastya/ingest_pipeline/)
try:
    from data_collector import main_collector, CORE_API_KEY, NEWSAPI_API_KEY
except ImportError:
    logging.critical("ERROR: Could not import 'main_collector' from data_collector.py.")
    logging.critical("Ensure data_collector.py is in the 'Agastya/ingest_pipeline/' directory.")
    exit()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INDICATIONS_FILENAME = "Indication Groupings.txt"
FINAL_OUTPUT_JSONL_FILENAME = "all_indications_raw_content.jsonl"
MAX_RESULTS_PER_SOURCE_PER_TERM = 300
DELAY_BETWEEN_TERMS_SECONDS = 4
MAX_LENGTH_TERM_FOR_GENERAL_APIS = 70

# --- Helper Function to Load Indications ---
def load_indications(resolved_indications_filepath): # DISTINCT PARAMETER NAME
    """Loads indication list from the JSON file (flat list)."""
    try:
        with open(resolved_indications_filepath, 'r', encoding='utf-8') as f: # USING DISTINCT PARAMETER
            data = json.load(f)
        if not isinstance(data, list):
            logging.error(f"Error: Expected a flat list of indications in {resolved_indications_filepath}, found {type(data)}")
            return []
        return [str(term).strip() for term in data if str(term).strip()]
    except FileNotFoundError:
        logging.error(f"Indication file not found at {resolved_indications_filepath}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {resolved_indications_filepath}")
        return []

# --- Main Orchestration ---
def run_full_content_ingestion(full_indications_path_arg, output_jsonl_full_path_arg): # DISTINCT ARG NAMES
    indication_terms = load_indications(full_indications_path_arg) # PASSING DISTINCT ARG
    if not indication_terms:
        logging.error("No indication terms loaded by load_indications. Exiting run_full_content_ingestion.")
        return

    all_accumulated_data = []
    unique_document_tracker = set() 

    logging.info(f"Starting content ingestion for {len(indication_terms)} indication terms from '{full_indications_path_arg}'.")
    logging.info(f"Max results per API per indication term: {MAX_RESULTS_PER_SOURCE_PER_TERM}")
    logging.info(f"Delay between indication terms: {DELAY_BETWEEN_TERMS_SECONDS}s")

    total_terms = len(indication_terms)
    for term_index, indication_term in enumerate(indication_terms):
        logging.info(f"Processing Indication Term {term_index + 1}/{total_terms}: '{indication_term}'")
        api_queries_for_current_term = {}
        api_queries_for_current_term["pubmed"] = indication_term
        api_queries_for_current_term["clinicaltrials"] = indication_term
        if CORE_API_KEY:
            api_queries_for_current_term["core"] = indication_term
        if NEWSAPI_API_KEY:
            if len(indication_term) > MAX_LENGTH_TERM_FOR_GENERAL_APIS:
                logging.debug(f"  Term '{indication_term}' potentially long for NewsAPI, using as is.")
            api_queries_for_current_term["newsapi"] = indication_term
        api_queries_for_current_term["who_news"] = indication_term
        api_queries_for_current_term["openfda"] = indication_term

        logging.debug(f"  API query map for term '{indication_term}': {api_queries_for_current_term}")
        try:
            collected_batch = main_collector(
                search_queries_map=api_queries_for_current_term,
                output_file=None, 
                max_results_per_source=MAX_RESULTS_PER_SOURCE_PER_TERM
            )
            if collected_batch:
                new_items_added_this_batch = 0
                for item in collected_batch:
                    doc_key = (item.get("source", "N/A").lower(), str(item.get("id", "N/A")).lower())
                    if (item.get("id", "N/A") != "N/A" and doc_key not in unique_document_tracker) or \
                       (item.get("id", "N/A") == "N/A"):
                        all_accumulated_data.append(item)
                        if item.get("id", "N/A") != "N/A":
                            unique_document_tracker.add(doc_key)
                        new_items_added_this_batch += 1
                logging.info(f"  Collected {new_items_added_this_batch} new items for term '{indication_term}'. Total accumulated: {len(all_accumulated_data)}")
            else:
                logging.info(f"  No data collected for term '{indication_term}'.")
        except Exception as e:
            logging.error(f"  ERROR during data collection for term '{indication_term}': {e}", exc_info=True)
        logging.info(f"  Pausing for {DELAY_BETWEEN_TERMS_SECONDS} seconds...")
        time.sleep(DELAY_BETWEEN_TERMS_SECONDS)

    logging.info(f"\nTotal unique items accumulated: {len(all_accumulated_data)}")
    if all_accumulated_data:
        valid_items_to_write_count = 0
        try:
            with open(output_jsonl_full_path_arg, 'w', encoding='utf-8') as f: # USING DISTINCT ARG
                for item in all_accumulated_data:
                    if item.get("text") and len(item.get("text").strip()) > 20:
                        f.write(json.dumps(item) + '\n')
                        valid_items_to_write_count +=1
            logging.info(f"Successfully wrote {valid_items_to_write_count} valid items to {output_jsonl_full_path_arg}")
        except IOError as e:
            logging.error(f"Error writing final data to {output_jsonl_full_path_arg}: {e}")
    else:
        logging.info("No data was accumulated to write to the final file.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"Loaded environment variables from: {dotenv_path}")
    else:
        logging.info(f".env file not found at {dotenv_path}, relying on system environment variables.")

    current_newsapi_key = os.getenv("NEWSAPI_API_KEY")
    current_core_api_key = os.getenv("CORE_API_KEY")
    if not current_newsapi_key:
        logging.warning("NEWSAPI_API_KEY is not set. NewsAPI calls will be skipped by data_collector.")
    if not current_core_api_key:
        logging.warning("CORE_API_KEY is not set. CORE API calls will be skipped by data_collector.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    indications_full_path = os.path.join(script_dir, INDICATIONS_FILENAME)
    output_jsonl_full_path = os.path.join(script_dir, FINAL_OUTPUT_JSONL_FILENAME)
    
    if not os.path.exists(indications_full_path):
        logging.error(f"CRITICAL: '{INDICATIONS_FILENAME}' not found at calculated path: {indications_full_path}")
        logging.error("Please ensure the file exists in the same directory as this script.")
    else:
        run_full_content_ingestion(indications_full_path, output_jsonl_full_path)