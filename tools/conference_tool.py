import os
import logging
import fitz  # PyMuPDF
import re
import json
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# --- Import Tavily for real web search ---
try:
    from tavily import TavilyClient
except ImportError:
    logging.error("Tavily SDK not found. Please install with: pip install tavily-python")

# --- Load environment variables for API keys ---
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Initialize Tavily client ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = None

if TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        logger.info("Tavily search client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Tavily client: {e}")
else:
    logger.warning("TAVILY_API_KEY not found in environment variables. Web search will not be available.")

# --- Import model configuration (if needed for future) ---
try:
    from agent.model_config import get_model_for_task
except ImportError:
    logger.warning("Could not import model configuration, using default models if needed")

# --- Define PDF file paths relative to the project root (Agastya/) ---
# This script (conference_tool.py) is in Agastya/tools/
# So, '..' goes up to Agastya/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Known medical conferences for better keyword matching
KNOWN_CONFERENCES = [
    "asco", "american society of clinical oncology",
    "esmo", "european society for medical oncology",
    "acc", "american college of cardiology",
    "ash", "american society of hematology",
    "aan", "american academy of neurology",
    "aha", "american heart association",
    "aacr", "american association for cancer research",
    "aua", "american urological association",
    "easl", "european association for the study of the liver"
]

PDF_FILES_INFO = {
    "past_conference": {
        "path": os.path.join(PROJECT_ROOT, "past_conference_summary.pdf"),
        "description": "a summary of a significant past medical conference (e.g., last year's ASCO or a major cardiology congress).",
        "keywords": ["past conference", "last year's conference", "previous meeting", "summary of"] # Add specific past conference names if known
    },
    "upcoming_conference": {
        "path": os.path.join(PROJECT_ROOT, "upcoming_conference_schedule.pdf"),
        "description": "information about an upcoming major medical conference (e.g., next year's ESMO or a key neurology event).",
        "keywords": ["upcoming conference", "next conference", "future meeting", "schedule for"] # Add specific upcoming conference names if known
    }
    # Add more PDFs here if needed, e.g., specific conference names mapping to specific files
    # "asco_2023": {
    #     "path": os.path.join(PROJECT_ROOT, "asco_2023_highlights.pdf"),
    #     "description": "highlights from ASCO 2023.",
    #     "keywords": ["asco 2023", "american society of clinical oncology 2023"]
    # }
}

def _extract_conference_name_from_query(query: str) -> str or None:
    """Extract specific conference name and year from user query if possible."""
    query_lower = query.lower()
    
    # Look for known conference names
    matched_conferences = []
    for conf in KNOWN_CONFERENCES:
        if conf in query_lower:
            matched_conferences.append(conf)
    
    # Find the longest match (to avoid partial matches)
    if matched_conferences:
        matched_conference = max(matched_conferences, key=len)
        
        # Try to extract year if present
        year_match = re.search(r'20\d{2}', query)
        year = year_match.group(0) if year_match else ""
        
        return f"{matched_conference} {year}".strip()
    
    return None

def _is_structured_output_needed(query: str) -> bool:
    """Detect if structured output is needed based on the query."""
    query_lower = query.lower()
    
    # Keywords that indicate the user wants structured information
    structured_indicators = [
        "structure", "structured", "format", "detailed", "details", "agenda", 
        "schedule", "program", "information about", "info about", "tell me about",
        "when and where", "registration info", "deadlines", "json", "data",
        "all the information", "comprehensive", "complete information"
    ]
    
    # Check if any indicators are in the query
    for indicator in structured_indicators:
        if indicator in query_lower:
            return True
    
    return False

def _extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extracts all text content from a given PDF file."""
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at: {pdf_path}")
        return None
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        logger.info(f"Successfully extracted text from {os.path.basename(pdf_path)}.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}", exc_info=True)
        return None

def _perform_tavily_search(query: str) -> Dict:
    """
    Performs an actual web search using Tavily API.
    Returns search results or an error message.
    """
    if not tavily_client:
        return {
            "success": False,
            "error": "Tavily client not initialized. Please set TAVILY_API_KEY environment variable.",
            "results": []
        }
    
    try:
        # Perform the search using Tavily API
        search_result = tavily_client.search(
            query=query,
            search_depth="advanced",  # Get more comprehensive results
            include_answer=True,      # Get a summarized answer
            include_domains=["asco.org", "esmo.org", "acc.org", "aacr.org", "aha.org", 
                           "aan.org", "easl.eu", "aua.org", "ash.org", "medscape.com", 
                           "medpagetoday.com", "healio.com"],  # Medical conference domains
            include_raw_content=False,  # We don't need the raw HTML
            max_results=5             # Limit to 5 results for relevance
        )
        
        return {
            "success": True,
            "search_id": search_result.get("search_id", ""),
            "answer": search_result.get("answer", ""),
            "results": search_result.get("results", [])
        }
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return {
            "success": False,
            "error": f"Error performing Tavily search: {str(e)}",
            "results": []
        }

def _search_web_for_conference(conference_name: str, user_query: str, structured_output: bool = False) -> Dict:
    """
    Uses Tavily to perform a real web search for conference information.
    Returns formatted results suitable for LLM consumption.
    """
    logger.info(f"Conference tool: Searching for '{conference_name}' using Tavily")
    
    # Create a targeted search query
    if structured_output:
        # For structured output, create a more comprehensive search query
        search_query = f"{conference_name} medical conference dates location venue registration deadlines agenda speakers abstract submission official website"
    elif "upcoming" in user_query.lower() or "next" in user_query.lower() or "future" in user_query.lower():
        search_query = f"{conference_name} medical conference upcoming dates location registration"
    elif "past" in user_query.lower() or "last" in user_query.lower() or "previous" in user_query.lower():
        search_query = f"{conference_name} medical conference past recent highlights summary"
    else:
        search_query = f"{conference_name} medical conference information dates schedule highlights"
    
    # Perform the actual search using Tavily
    search_results = _perform_tavily_search(search_query)
    
    if not search_results["success"]:
        # If search fails, return error but format as web_search so the LLM knows what happened
        return {
            "source": "web_search",
            "success": False,
            "search_query": search_query,
            "description": f"Web search for {conference_name} (failed)",
            "content": f"Error performing web search: {search_results.get('error', 'Unknown error')}"
        }
    
    # Format the results for the LLM
    formatted_content = ""
    
    # Add the Tavily-generated answer if available
    if search_results.get("answer"):
        formatted_content += f"SEARCH SUMMARY: {search_results['answer']}\n\n"
    
    # Add individual search results
    formatted_content += "SEARCH RESULTS:\n\n"
    for i, result in enumerate(search_results.get("results", []), 1):
        formatted_content += f"[{i}] {result.get('title', 'No title')}\n"
        formatted_content += f"URL: {result.get('url', 'No URL')}\n"
        formatted_content += f"Date: {result.get('published_date', 'Unknown date')}\n"
        formatted_content += f"Content: {result.get('content', 'No content')}\n\n"
    
    return {
        "source": "web_search",
        "success": True,
        "search_query": search_query,
        "structured_output": structured_output,
        "search_id": search_results.get("search_id", ""),
        "description": f"Web search results about {conference_name}",
        "content": formatted_content
    }

def get_conference_document_content(user_query: str) -> dict | None:
    """
    Gets conference information based on the user query.
    First tries to search the web with Tavily if available,
    then falls back to static PDF documents if needed.

    Returns a dictionary with source information and content.
    """
    # Always use structured output format (bulleted) for all conference queries
    structured_output = True
    
    # Try to extract a specific conference name from the query
    conference_name = _extract_conference_name_from_query(user_query)
    
    if conference_name and tavily_client:
        # We found a specific conference and Tavily is available, do a real web search
        logger.info(f"Conference tool: Detected specific conference '{conference_name}' in query")
        return _search_web_for_conference(conference_name, user_query, structured_output)
    elif conference_name and not tavily_client:
        # We found a specific conference but can't do web search, inform the user
        logger.warning(f"Conference tool: Detected conference '{conference_name}' but Tavily is not available")
        return {
            "source": "web_search",
            "success": False,
            "search_query": f"{conference_name} medical conference information",
            "description": f"Web search for {conference_name} (unavailable)",
            "content": "Web search is not available because Tavily API is not configured. Please add TAVILY_API_KEY to your environment variables to enable web search capabilities."
        }
    
    # No specific conference identified, check for general conference terms
    query_lower = user_query.lower()
    
    # Check for general keywords about conferences
    if any(term in query_lower for term in ["conference", "meeting", "congress", "symposium"]) and tavily_client:
        # Generic conference query, use Tavily for medical conferences search
        if structured_output:
            search_query = "upcoming major medical conferences healthcare 2024 2025 dates locations registration agenda"
        else:
            search_query = "upcoming major medical conferences healthcare dates locations"
            
        search_results = _perform_tavily_search(search_query)
        
        if search_results["success"]:
            formatted_content = "SEARCH SUMMARY: "
            if search_results.get("answer"):
                formatted_content += f"{search_results['answer']}\n\n"
            else:
                formatted_content += "Information about upcoming and recent major medical conferences.\n\n"
                
            formatted_content += "SEARCH RESULTS:\n\n"
            for i, result in enumerate(search_results.get("results", []), 1):
                formatted_content += f"[{i}] {result.get('title', 'No title')}\n"
                formatted_content += f"URL: {result.get('url', 'No URL')}\n"
                formatted_content += f"Date: {result.get('published_date', 'Unknown date')}\n"
                formatted_content += f"Content: {result.get('content', 'No content')}\n\n"
                
            return {
                "source": "web_search",
                "success": True,
                "search_query": search_query,
                "structured_output": structured_output,
                "description": "Information about major medical conferences",
                "content": formatted_content
            }
    
    # Attempt to identify the relevant PDF based on keywords as fallback
    selected_pdf_key = None
    highest_keyword_match_count = 0

    for pdf_key, info in PDF_FILES_INFO.items():
        current_match_count = 0
        for keyword in info["keywords"]:
            if keyword.lower() in query_lower:
                current_match_count +=1
        
        # Prioritize if more keywords match or if a specific PDF key is in the query
        if pdf_key.replace("_", " ") in query_lower: # e.g. if user says "asco 2023"
            current_match_count += 10 # Heavily weight direct name match
            
        if current_match_count > highest_keyword_match_count:
            highest_keyword_match_count = current_match_count
            selected_pdf_key = pdf_key

    if not selected_pdf_key: # Default or fallback if no strong keyword match
        # You could default to upcoming, or try to infer from words like "next", "last"
        if "upcoming" in query_lower or "next" in query_lower or "future" in query_lower:
            selected_pdf_key = "upcoming_conference"
        elif "past" in query_lower or "last" in query_lower or "previous" in query_lower or "summary of" in query_lower:
            selected_pdf_key = "past_conference"
        else:
            logger.warning(f"Conference tool: Could not determine relevant conference PDF for query: '{user_query}'. No specific keywords matched well.")
            # If we have Tavily, try a generic search as last resort
            if tavily_client:
                if structured_output:
                    search_query = "major medical conferences for healthcare professionals upcoming 2024 2025 dates locations agenda registration"
                else:
                    search_query = "major medical conferences for healthcare professionals upcoming"
                    
                search_results = _perform_tavily_search(search_query)
                
                if search_results["success"]:
                    formatted_content = "SEARCH SUMMARY: "
                    if search_results.get("answer"):
                        formatted_content += f"{search_results['answer']}\n\n"
                    else:
                        formatted_content += "General information about medical conferences.\n\n"
                        
                    formatted_content += "SEARCH RESULTS:\n\n"
                    for i, result in enumerate(search_results.get("results", []), 1):
                        formatted_content += f"[{i}] {result.get('title', 'No title')}\n"
                        formatted_content += f"URL: {result.get('url', 'No URL')}\n"
                        formatted_content += f"Content: {result.get('content', 'No content')}\n\n"
                        
                    return {
                        "source": "web_search",
                        "success": True,
                        "search_query": search_query,
                        "structured_output": structured_output,
                        "description": "Information about medical conferences",
                        "content": formatted_content
                    }
            # If no Tavily or search failed, default to upcoming
            selected_pdf_key = "upcoming_conference"

    if selected_pdf_key and selected_pdf_key in PDF_FILES_INFO:
        pdf_info = PDF_FILES_INFO[selected_pdf_key]
        logger.info(f"Conference tool: Selected PDF '{selected_pdf_key}' for query '{user_query}'.")
        content = _extract_text_from_pdf(pdf_info["path"])
        if content:
            return {
                "source": "pdf",
                "pdf_name": selected_pdf_key,
                "description": pdf_info["description"], # Description of what the PDF contains
                "content": content # The full text content of the PDF
            }
        else:
            return { # Return info about the PDF even if content extraction failed, LLM can say it couldn't read it
                "source": "pdf",
                "pdf_name": selected_pdf_key,
                "description": pdf_info["description"],
                "content": f"Error: Could not extract content from the PDF '{os.path.basename(pdf_info['path'])}'."
            }
    
    logger.warning(f"Conference tool: No suitable conference document found for query: '{user_query}'")
    return None


# --- Example Test (for direct testing of this file) ---
if __name__ == "__main__":
    # Test with various queries
    test_queries = [
        "Tell me about the upcoming ASCO 2024 conference.",
        "What was discussed at ESMO 2023?",
        "Give me information about ACC conference.",
        "When is the next AHA meeting?",
        "Tell me about the upcoming conference.",
        "What was discussed at the last major medical meeting?",
        "Any news from recent oncology conferences?"
    ]

    for q in test_queries:
        print(f"\n--- User Query: {q} ---")
        conference_data = get_conference_document_content(q)
        if conference_data:
            print(f"Source: {conference_data.get('source', 'unknown')}")
            if conference_data.get('source') == 'web_search':
                print(f"Search Success: {conference_data.get('success', False)}")
                print(f"Search Query: {conference_data.get('search_query', '')}")
                print(f"Description: {conference_data.get('description', '')}")
                print(f"Content (first 500 chars): {conference_data.get('content', '')[:500]}...")
            else:
                print(f"Selected PDF: {conference_data.get('pdf_name')}")
                print(f"PDF Description: {conference_data.get('description')}")
                print(f"Content (first 500 chars):\n{conference_data.get('content', '')[:500]}...")
        else:
            print("No relevant conference information found.")