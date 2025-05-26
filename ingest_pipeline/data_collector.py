import requests
import json
import os
from datetime import datetime, timedelta
import logging
import time
from urllib.parse import quote

# --- 0. Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Keys (Store in environment variables)
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# Base URLs - Verified against general knowledge and common API structures
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NEWSAPI_BASE_URL = "https://newsapi.org/v2/"
WHO_NEWS_BASE_URL = "https://www.who.int/api/news/newsitems" # As per your table & common use
CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2/" # As per your table
OPENFDA_BASE_URL = "https://api.fda.gov/"

# --- 1. Helper Function for API Requests ---
def make_api_request(url, params=None, headers=None, method="GET", data=None, max_retries=3, delay=5, is_text_response=False):
    """Generic function to make an API request and handle basic errors with retries."""
    for attempt in range(max_retries):
        try:
            if method.upper() == "POST":
                response = requests.post(url, params=params, headers=headers, json=data, timeout=30)
            else: # Default to GET
                response = requests.get(url, params=params, headers=headers, timeout=30)
            
            response.raise_for_status()
            if response.status_code == 204:
                return None
            return response.text if is_text_response else response.json() # Handle text or JSON
        except requests.exceptions.HTTPError as http_err:
            response_text_snippet = http_err.response.text[:200] if http_err.response else 'No response body'
            logging.error(f"HTTP error: {http_err} - URL: {url} - Attempt {attempt+1} - Resp: {response_text_snippet}")
            if hasattr(http_err, 'response') and http_err.response is not None:
                if http_err.response.status_code == 429: 
                    logging.warning(f"Rate limit. Waiting {delay * (attempt + 1)}s.")
                    time.sleep(delay * (attempt + 1))
                elif 500 <= http_err.response.status_code < 600: time.sleep(delay * (attempt + 1))
                else: break
            else: time.sleep(delay * (attempt + 1))
        except requests.exceptions.ConnectionError as err: logging.error(f"Conn error: {err}-URL:{url}-Attempt {attempt+1}"); time.sleep(delay * (attempt+1))
        except requests.exceptions.Timeout as err: logging.error(f"Timeout: {err}-URL:{url}-Attempt {attempt+1}"); time.sleep(delay * (attempt+1))
        except requests.exceptions.RequestException as err: logging.error(f"Request error: {err}-URL:{url}-Attempt {attempt+1}"); break
        except json.JSONDecodeError as err:
            resp_text = locals().get('response', None)
            logging.error(f"JSON decode error: {err}-URL:{url}-Resp: {resp_text.text[:200] if resp_text else 'N/A'}"); break
    return None

# --- 2. Standardized Output Formatter (Using "text" for main body as per previous design) ---
def process_to_standard_format(item_id, source_name, title, text_content, authors_list, publication_date_str, item_url, additional_metadata=None):
    if additional_metadata is None: additional_metadata = {}
    if not isinstance(text_content, str): text_content = str(text_content) if text_content is not None else ""
    if not isinstance(authors_list, list): authors_list = [str(authors_list)] if authors_list else []
    
    # Attempt to parse and reformat publication_date_str to YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ
    parsed_date = None
    if publication_date_str:
        try: # Try ISO format first (common from APIs)
            dt_obj = datetime.fromisoformat(publication_date_str.replace('Z', '+00:00'))
            parsed_date = dt_obj.strftime('%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            try: # Try YYYYMMDD (like from openFDA)
                dt_obj = datetime.strptime(publication_date_str, '%Y%m%d')
                parsed_date = dt_obj.strftime('%Y-%m-%d')
            except ValueError:
                 try: # Try YYYY-MM-DD (already good)
                    datetime.strptime(publication_date_str, '%Y-%m-%d')
                    parsed_date = publication_date_str
                 except ValueError:
                    try: # Try just year (e.g., PubMed sometimes)
                        dt_obj = datetime.strptime(publication_date_str, '%Y')
                        parsed_date = dt_obj.strftime('%Y-%m-%d') # Default to Jan 1st
                    except ValueError:
                        logging.warning(f"Could not parse date: {publication_date_str} for source {source_name}. Leaving as is or None.")
                        parsed_date = publication_date_str # Fallback to original if unparseable but not None

    return {
        "id": str(item_id).strip() if item_id else "N/A",
        "source": source_name.strip() if source_name else "N/A",
        "title": title.strip() if title else "N/A",
        "text": text_content.strip(), # Maps to "body" in your table
        "authors": authors_list,
        "publication_date": parsed_date,
        "url": item_url.strip() if item_url else "N/A",
        "retrieved_at": datetime.utcnow().isoformat() + "Z",
        "metadata": additional_metadata
    }

# --- 3. Fetcher and Processor Functions ---

def fetch_pubmed_data(search_term, max_results=50, start_date_str=None, end_date_str=None):
    logging.info(f"Fetching PubMed data for: {search_term} (Dates: {start_date_str} to {end_date_str})")
    processed_articles = []
    # 1. ESearch to get PMIDs
    date_query_part = ""
    if start_date_str and end_date_str:
        # Assuming YYYY-MM-DD format for input, convert to YYYY/MM/DD for PubMed
        try:
            s_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y/%m/%d")
            e_date = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%Y/%m/%d")
            date_query_part = f" AND ({s_date}[PDAT] : {e_date}[PDAT])"
        except ValueError:
            logging.warning(f"Invalid date format for PubMed. Expected YYYY-MM-DD. Dates: {start_date_str}, {end_date_str}")
    elif start_date_str: # Only start date
        try:
            s_date = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y/%m/%d")
            date_query_part = f" AND ({s_date}[PDAT] : {datetime.now().strftime('%Y/%m/%d')}[PDAT])" # Search from start_date to now
        except ValueError:
            logging.warning(f"Invalid start date format for PubMed: {start_date_str}")
            
    effective_search_term = search_term + date_query_part

    search_url = f"{PUBMED_BASE_URL}esearch.fcgi"
    search_params = {
        "db": "pubmed", "term": effective_search_term, "retmax": max_results, # Use effective_search_term
        "usehistory": "y", "format": "json", "api_key": PUBMED_API_KEY
    }
    search_response = make_api_request(search_url, params=search_params)
    if not search_response or "esearchresult" not in search_response or "idlist" not in search_response["esearchresult"]:
        logging.error(f"PubMed ESearch failed for '{search_term}'. Resp: {str(search_response)[:200]}")
        return processed_articles
    ids = search_response["esearchresult"]["idlist"]
    if not ids:
        logging.info(f"No PubMed PMIDs found for '{search_term}'.")
        return processed_articles

    # 2. For each PMID, get ESummary for metadata and EFetch for abstract text
    for pmid in ids:
        # Get ESummary for metadata
        summary_url = f"{PUBMED_BASE_URL}esummary.fcgi"
        summary_params = {"db": "pubmed", "id": pmid, "format": "json", "api_key": PUBMED_API_KEY}
        summary_data_response = make_api_request(summary_url, params=summary_params)
        
        article_metadata = None
        if summary_data_response and "result" in summary_data_response and pmid in summary_data_response["result"]:
            article_metadata = summary_data_response["result"][pmid]
        
        if not article_metadata:
            logging.warning(f"Could not retrieve ESummary metadata for PMID {pmid}.")
            continue

        title = article_metadata.get("title", "N/A")
        pub_date_str = article_metadata.get("pubdate", article_metadata.get("epubdate", "")) # Format YYYY, YYYY Mon, YYYY Mon DD
        authors_list = [author.get("name") for author in article_metadata.get("authors", []) if author.get("name")]
        journal_name = article_metadata.get("fulljournalname", "")
        doi = next((aid['value'] for aid in article_metadata.get('articleids', []) if aid.get('idtype') == 'doi'), None)
        
        # Get EFetch for abstract text
        abstract_text = "Abstract not available."
        fetch_url = f"{PUBMED_BASE_URL}efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text", "api_key": PUBMED_API_KEY}
        fetched_abstract = make_api_request(fetch_url, params=fetch_params, is_text_response=True)
        if fetched_abstract and isinstance(fetched_abstract, str) and fetched_abstract.strip():
            # Clean up common EFetch text abstract prefixes/formatting if any
            abstract_text = fetched_abstract.strip() 
            # Remove potential repeated title at the start of abstract if EFetch includes it
            if abstract_text.lower().startswith(title.lower()):
                abstract_text = abstract_text[len(title):].lstrip('.\n ')


        item_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        metadata = {"pmid": pmid, "doi": doi, "journal": journal_name}
        
        processed_articles.append(
            process_to_standard_format(pmid, "PubMed", title, abstract_text, authors_list, pub_date_str, item_url, metadata)
        )
        time.sleep(0.4) # Be nice to NCBI Entrez Utilities (3 requests per second without API key, 10 with)

    logging.info(f"Fetched {len(processed_articles)} items from PubMed for '{effective_search_term}'.")
    return processed_articles

def fetch_newsapi_data(search_term, category="health", country="us", max_results=100, start_date_str=None, end_date_str=None):
    logging.info(f"Fetching NewsAPI data for: {search_term} (category: {category}, country: {country}, dates: {start_date_str} to {end_date_str})")
    processed_articles = []
    if not NEWSAPI_API_KEY:
        logging.warning("NEWSAPI_API_KEY not set. Skipping NewsAPI.")
        return processed_articles

    # Change from top-headlines to everything endpoint to support date filtering
    url = f"{NEWSAPI_BASE_URL}everything" 
    params = {
        "q": search_term, 
        "pageSize": max_results, 
        "apiKey": NEWSAPI_API_KEY,
        "language": "en"
    }
    
    # Add date range parameters only if dates are provided
    if start_date_str:
        params["from"] = start_date_str
    if end_date_str:
        params["to"] = end_date_str
    
    response_data = make_api_request(url, params=params)
    if response_data and "articles" in response_data:
        for article in response_data["articles"]:
            title = article.get("title", "N/A")
            # "body":"<description>"
            text_content = article.get("description", "")
            if not text_content: # Fallback to content snippet if description is empty
                text_content = article.get("content", "")
            if text_content and "[+" in text_content: # Remove "[+1234 chars]"
                text_content = text_content.split("[+")[0].strip()
            if not text_content and title != "N/A": text_content = title # Ultimate fallback if no body text

            authors_list = [article.get("author")] if article.get("author") else [] # Table: ["Author Name"]
            publication_date_str = article.get("publishedAt") # ISO 8601 format YYYY-MM-DDTHH:MM:SSZ
            article_url = article.get("url")
            
            source_info = article.get("source", {})
            source_name_api = source_info.get("name", "UnknownSource")
            # Use "NewsAPI" as the main source, and actual source name in metadata
            metadata = {"news_source_id": source_info.get("id"), "news_source_name": source_name_api}

            processed_articles.append(
                process_to_standard_format(article_url, "NewsAPI", title, text_content, authors_list, publication_date_str, article_url, metadata)
            )
    else:
        logging.warning(f"No 'articles' in NewsAPI response for '{search_term}'. Response: {str(response_data)[:200]}")
    logging.info(f"Fetched {len(processed_articles)} articles from NewsAPI for '{search_term}'.")
    return processed_articles

def fetch_who_news_data(search_term=None, max_results=100, start_date_str=None, end_date_str=None):
    logging.info(f"Fetching WHO News data (max: {max_results}, dates: {start_date_str} to {end_date_str}). Search term: '{search_term}'.")
    processed_items = []
    # Endpoint: GET /api/news/newsitems
    params = {"$orderby": "PublicationDateUtc desc", "$top": max_results}
    
    # Build OData filter for search and date range
    filter_parts = []
    
    # Add date range filter if dates are provided
    if start_date_str and end_date_str:
        filter_parts.append(f"PublicationDateAndTime ge {start_date_str}T00:00:00Z and PublicationDateAndTime le {end_date_str}T23:59:59Z")
    
    # Add search term filter if provided
    if search_term:
        filter_parts.append(f"contains(tolower(Title),tolower('{search_term}')) or contains(tolower(Content),tolower('{search_term}'))")
    
    # Combine filter parts with 'and' if we have multiple conditions
    if filter_parts:
        params["$filter"] = " and ".join(filter_parts)
        
    response_data = make_api_request(WHO_NEWS_BASE_URL, params=params)
    
    items_to_process = []
    if response_data and isinstance(response_data, list): items_to_process = response_data # Direct list
    elif response_data and "value" in response_data and isinstance(response_data["value"], list): items_to_process = response_data["value"] # OData envelope
    else: logging.warning(f"WHO News API unexpected format/no data for '{search_term}'. Resp: {str(response_data)[:200]}")

    for item_index, item in enumerate(items_to_process):
        # ID: SystemSourceKey (ID from table)
        item_id = item.get("SystemSourceKey", item.get("Id", f"who_item_{search_term or 'latest'}_{item_index}"))
        title = item.get("Title", "N/A")
        # "body":"...(if full text fetched)..." - API provides 'Content' which is likely summary
        text_content = item.get("Content", item.get("Summary", "")) 
        
        authors_list = ["World Health Organization"] # As per table
        # Date: PublicationDateAndTime or PublicationDateUtc
        publication_date_str = item.get("PublicationDateAndTime", item.get("PublicationDateUtc")) 
        
        # URL: ItemDefaultUrl or construct from UrlName
        item_url = item.get("ItemDefaultUrl") 
        if not item_url and item.get("UrlName"):
            item_url = f"https://www.who.int/news/item/{item.get('UrlName')}" # Constructing, may need adjustment
        if not item_url: item_url = item.get("Link") # Fallback

        metadata = {"guid": item.get("GUID", item.get("Id")), "topic_tags": item.get("TopicTags")}
        processed_items.append(
            process_to_standard_format(item_id, "WHO News", title, text_content, authors_list, publication_date_str, item_url, metadata)
        )
    logging.info(f"Fetched {len(processed_items)} items from WHO News for '{search_term}'.")
    return processed_items

def fetch_clinicaltrials_data(search_term, max_results=30, start_date_str=None, end_date_str=None):
    logging.info(f"Fetching ClinicalTrials.gov data for: {search_term} (dates: {start_date_str} to {end_date_str})")
    processed_trials = []
    # Endpoint: GET /api/v2/studies?query=<terms>&status=<Recruiting>&pageSize=...
    url = f"{CLINICALTRIALS_BASE_URL}studies"
    params = {"query.term": search_term, "pageSize": max_results, "format": "json"} # Add other filters like status if needed
    
    # Add date filtering using lastUpdatePosted
    if start_date_str and end_date_str:
        # Convert YYYY-MM-DD to YYYYMMDD for ClinicalTrials.gov API
        start_date_ct = start_date_str.replace("-", "")
        end_date_ct = end_date_str.replace("-", "")
        params["query.filter"] = f"lastUpdatePosted RANGE[{start_date_ct}, {end_date_ct}]"
    
    response_data = make_api_request(url, params=params)
    if response_data and "studies" in response_data:
        for study_wrapper in response_data["studies"]:
            study_protocol = study_wrapper.get("protocolSection", {})
            if not study_protocol: continue

            id_module = study_protocol.get("identificationModule", {})
            nct_id = id_module.get("nctId", "N/A")
            # Title: briefTitle (as per table) or officialTitle
            title = id_module.get("briefTitle", id_module.get("officialTitle", "N/A"))
            
            desc_module = study_protocol.get("descriptionModule", {})
            # "body":"...briefSummary or detailedDescription..."
            text_content = desc_module.get("briefSummary", "")
            if not text_content: text_content = desc_module.get("detailedDescription", "")
            # Optionally add eligibilityCriteria if desired for more complete text
            # eligibility_criteria = study_protocol.get("eligibilityModule", {}).get("eligibilityCriteria", "")
            # if eligibility_criteria: text_content += f"\n\nEligibility Criteria: {eligibility_criteria}"

            # Authors: ["Sponsor Name or PI"]
            authors_list = []
            sponsor_module = study_protocol.get("sponsorCollaboratorsModule", {})
            if sponsor_module and sponsor_module.get("leadSponsor", {}).get("name"):
                authors_list.append(sponsor_module["leadSponsor"]["name"])
            # Could also check responsiblePartyModule for PI if leadSponsor isn't enough
            # resp_party_module = study_protocol.get("responsiblePartyModule", {})
            # if resp_party_module and resp_party_module.get("type") == "PRINCIPAL_INVESTIGATOR" and resp_party_module.get("investigatorFullName"):
            #    if not authors_list or resp_party_module.get("investigatorFullName") not in authors_list: # Avoid duplicate if sponsor is PI
            #        authors_list.append(resp_party_module["investigatorFullName"])
            if not authors_list: authors_list.append("Sponsor/PI not specified")


            status_module = study_protocol.get("statusModule", {})
            # Date: YYYY-MM-DD (lastUpdatePosted or startDate)
            publication_date_str = status_module.get("lastUpdatePostDateStruct", {}).get("date")
            if not publication_date_str:
                publication_date_str = status_module.get("startDateStruct", {}).get("date")
            
            # URL: https://clinicaltrials.gov/ct2/show/<NCT_ID> (as per table)
            # The API v2 documentation uses /study/NCT_ID as the canonical link
            item_url = f"https://clinicaltrials.gov/study/{nct_id}"
            # item_url = f"https://www.clinicaltrials.gov/ct2/show/{nct_id}" # Old link structure
            
            metadata = {
                "nctId": nct_id, "status": status_module.get("overallStatus"),
                "studyType": study_protocol.get("designModule", {}).get("studyType"),
                "phases": study_protocol.get("designModule", {}).get("phases"),
                "conditions": study_protocol.get("conditionsModule", {}).get("conditions", []),
                "interventions": [iv.get("name") for iv in study_protocol.get("armsInterventionsModule", {}).get("interventions", []) if iv.get("name")]
            }
            processed_trials.append(
                process_to_standard_format(nct_id, "ClinicalTrials.gov", title, text_content, authors_list, publication_date_str, item_url, metadata)
            )
    else:
        logging.warning(f"No 'studies' in ClinicalTrials.gov response for '{search_term}'. Response: {str(response_data)[:200]}")
    logging.info(f"Fetched {len(processed_trials)} trials from ClinicalTrials.gov for '{search_term}'.")
    return processed_trials

def fetch_openfda_data(search_term, endpoint_config=None, limit=30, start_date_str=None, end_date_str=None):
    # Default endpoint config if none provided
    if endpoint_config is None:
        endpoint_config = {"endpoint": "drug/event.json", "source_name_suffix": "drug-event", "title_prefix": "Adverse Event Report"}

    endpoint = endpoint_config.get("endpoint", "drug/event.json")
    source_name = f"openFDA-{endpoint_config.get('source_name_suffix', endpoint.split('/')[0])}"
    title_prefix_template = endpoint_config.get("title_prefix", "OpenFDA Report")
    
    logging.info(f"Fetching {source_name} data for term: '{search_term}' (dates: {start_date_str} to {end_date_str})")
    processed_items = []
    url = f"{OPENFDA_BASE_URL}{endpoint}"
    
    # General search query. Specific endpoints might need more structured queries.
    # For "drug/event" search in relevant fields:
    if endpoint == "drug/event.json":
        base_query = f'(patient.drug.drugindication:"{search_term}" OR patient.reaction.reactionmeddrapt:"{search_term}" OR openfda.brand_name:"{search_term}" OR openfda.generic_name:"{search_term}")'
        # Add date range for drug/event
        if start_date_str and end_date_str:
            date_query = f'+AND+receivedate:[{start_date_str}+TO+{end_date_str}]'
            query_string = base_query + date_query
        else:
            query_string = base_query
    elif endpoint == "drug/label.json": # Example for drug labels
        base_query = f'(openfda.brand_name:"{search_term}" OR openfda.generic_name:"{search_term}" OR indications_and_usage:"{search_term}")'
        # Add date range for drug/label
        if start_date_str and end_date_str:
            date_query = f'+AND+effective_time:[{start_date_str}+TO+{end_date_str}]'
            query_string = base_query + date_query
        else:
            query_string = base_query
    elif endpoint == "drug/drugsfda.json": # Example for FDA approvals
        base_query = f'(products.brand_name:"{search_term}" OR sponsor_name:"{search_term}")'
        # Add date range for drugsfda using submission_status_date
        if start_date_str and end_date_str:
            date_query = f'+AND+submissions.submission_status_date:[{start_date_str}+TO+{end_date_str}]'
            query_string = base_query + date_query
        else:
            query_string = base_query
    else: # Fallback to general quoted search term
        base_query = f'"{search_term}"'
        # Generic date filtering - may need to be adjusted based on actual endpoint
        if start_date_str and end_date_str:
            date_query = f'+AND+effective_time:[{start_date_str}+TO+{end_date_str}]'
            query_string = base_query + date_query
        else:
            query_string = base_query

    params = {"search": query_string, "limit": limit}
    response_data = make_api_request(url, params=params)

    if response_data and "results" in response_data:
        for item_index, item in enumerate(response_data["results"]):
            item_id, title, text_content, publication_date_str = f"{source_name}_{search_term}_{item_index}", "N/A", "", None
            authors_list = ["FDA"] # As per table

            if endpoint == "drug/event.json":
                item_id = item.get("safetyreportid", item_id)
                title = f"{title_prefix_template} {item_id}"
                text_parts = []
                if item.get("patient"):
                    reactions = [r.get("reactionmeddrapt") for r in item.get("patient", {}).get("reaction",[]) if r.get("reactionmeddrapt")]
                    if reactions: text_parts.append(f"Reactions: {', '.join(reactions)}")
                    drugs = [d.get("medicinalproduct") for d in item.get("patient", {}).get("drug",[]) if d.get("medicinalproduct")]
                    if drugs: text_parts.append(f"Drugs: {', '.join(drugs)}")
                text_content = ". ".join(filter(None, text_parts))
                if not text_content: text_content = "Adverse event summary." # Fallback
                date_val = item.get("receiptdate") # YYYYMMDD
                if date_val and len(date_val) == 8: publication_date_str = date_val

            elif endpoint == "drug/label.json": # Example for drug labels
                item_id = item.get("id", item_id)
                brand_names = item.get("openfda",{}).get("brand_name", [])
                title = f"Drug Label: {brand_names[0] if brand_names else search_term}"
                text_content = ". ".join(item.get("indications_and_usage", []) + item.get("warnings", [])) # Example text from label
                if not text_content: text_content = "Drug label information."
                date_val = item.get("effective_time") # YYYYMMDD
                if date_val and len(date_val) == 8: publication_date_str = date_val
            
            elif endpoint == "drug/drugsfda.json": # Example for drug approvals
                item_id = item.get("application_number", item_id)
                brand_name = ""
                if item.get("products"): brand_name = item["products"][0].get("brand_name", "")
                title = f"FDA Approval - {brand_name if brand_name else search_term}" # As per table
                text_content = f"Sponsor: {item.get('sponsor_name','')}. Approval for {brand_name}." # Simple summary
                # Find an approval date
                if item.get("submissions"):
                    approval_dates = [sub.get("submission_status_date") for sub in item["submissions"] if sub.get("submission_status") == "AP"]
                    if approval_dates: publication_date_str = sorted(approval_dates)[-1] # Get latest approval date (YYYYMMDD format)
            
            else: # Generic fallback for other openFDA endpoints
                item_id = item.get("id", item.get(list(item.keys())[0] if item.keys() else "id_fallback", item_id)) # Try to get some ID
                title = f"{title_prefix_template} for '{search_term}' - Item {item_index+1}"
                text_content = json.dumps(item, indent=2, sort_keys=True)[:1500] # Truncated JSON dump
                # Try to find any date field
                date_keys = ['report_date', 'created_date', 'date', 'effective_time', 'last_update_posted']
                for dk in date_keys:
                    if item.get(dk): publication_date_str = item.get(dk); break

            # URL: <FDA webpage or API reference> - Constructing a query link
            item_url = f"https://open.fda.gov/apis/{endpoint.replace('.json', '')}/explore-the-api-with-the-search-tester/?search={requests.utils.quote(query_string)}&limit={limit}"
            
            additional_md = {"original_query": query_string, "openfda_endpoint": endpoint}
            if endpoint == "drug/drugsfda.json" and item.get("application_number"):
                additional_md["application_number"] = item["application_number"]

            processed_items.append(
                process_to_standard_format(item_id, source_name, title, text_content, authors_list, publication_date_str, item_url, additional_md)
            )
    else:
        logging.warning(f"No 'results' in {source_name} response for query '{query_string}'. Response: {str(response_data)[:200]}")
    logging.info(f"Fetched {len(processed_items)} items from {source_name} for '{search_term}'.")
    return processed_items


# --- 4. Main Collector Orchestration ---
def main_collector(search_queries_map, output_file="collected_hcp_data.jsonl", max_results_per_source=50):
    all_collected_data = []
    logging.info(f"Data Collector: Processing queries: {search_queries_map}. Max results/source: {max_results_per_source}")

    if "pubmed" in search_queries_map and search_queries_map["pubmed"]:
        all_collected_data.extend(fetch_pubmed_data(search_queries_map["pubmed"], max_results=max_results_per_source))
    
    # openFDA - fetch_openfda_data now takes an endpoint_config
    # For demonstration, let's assume the orchestrator wants drug events and drug approvals for a term
    if "openfda" in search_queries_map and search_queries_map["openfda"]:
        openfda_term = search_queries_map["openfda"]
        all_collected_data.extend(fetch_openfda_data(openfda_term, 
            endpoint_config={"endpoint": "drug/event.json", "source_name_suffix": "drug-event", "title_prefix": "Adverse Event"}, 
            limit=max_results_per_source))
        all_collected_data.extend(fetch_openfda_data(openfda_term, 
            endpoint_config={"endpoint": "drug/drugsfda.json", "source_name_suffix": "drug-approval", "title_prefix": "FDA Drug Approval"},
            limit=max_results_per_source))
        # Could add more openFDA endpoints like drug/label.json if desired

    if "newsapi" in search_queries_map and search_queries_map["newsapi"]:
        all_collected_data.extend(fetch_newsapi_data(search_queries_map["newsapi"], max_results=max_results_per_source))
    
    if "who_news" in search_queries_map: # search_term can be None
        all_collected_data.extend(fetch_who_news_data(search_queries_map["who_news"], max_results=max_results_per_source))
    
    if "clinicaltrials" in search_queries_map and search_queries_map["clinicaltrials"]:
        all_collected_data.extend(fetch_clinicaltrials_data(search_queries_map["clinicaltrials"], max_results=max_results_per_source))

    if output_file:
        valid_items_count = 0
        try:
            with open(output_file, 'w', encoding='utf-8') as f: 
                for item in all_collected_data:
                    if item.get("text") and len(item.get("text").strip()) > 10:
                        f.write(json.dumps(item) + '\n')
                        valid_items_count += 1
            logging.info(f"Data Collector: Wrote {valid_items_count} valid items to {output_file}")
        except IOError as e:
            logging.error(f"Data Collector: Error writing to output file {output_file}: {e}")
    else:
        logging.info(f"Data Collector: Output file not specified. Returning {len(all_collected_data)} items.")
    return all_collected_data

if __name__ == "__main__":
    logging.info("Testing data_collector.py directly with updated fetchers...")
    test_queries = {
        "pubmed": "aspirin headache",
        "newsapi": "health technology",
        "who_news": "ebola", # Test with a term
        # "who_news": None, # Test for latest
        "clinicaltrials": "nivolumab lung cancer",
        "openfda": "atorvastatin" # General term for openFDA endpoints
    }
    test_data = main_collector(test_queries, output_file="test_direct_collection_v2.jsonl", max_results_per_source=2)
    logging.info(f"Direct test collected {len(test_data)} items.")

    # Test PubMed specifically for abstract fetching
    # test_pubmed_alone = fetch_pubmed_data("covid-19 vaccine", max_results=1)
    # if test_pubmed_alone:
    #     logging.info(f"PubMed specific test result: {json.dumps(test_pubmed_alone[0], indent=2)}")