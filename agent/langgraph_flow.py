import os
import logging
from typing import TypedDict, Annotated, List, Union, Optional, Dict
import operator
import json # For formatting some metadata if needed
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- Import model configuration ---
from .model_config import get_model_for_task

# --- .env loading (crucial for API keys) ---
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"LangGraph Flow: Loaded .env from {dotenv_path}")
else:
    logging.info(f"LangGraph Flow: .env not found at {dotenv_path}, relying on system vars.")
# --- END .env loading ---

# --- Import Tool Functions ---
try:
    from tools.research_tool import query_chroma_db
    from tools.panel_tool import handle_panel_query
    from tools.conference_tool import get_conference_document_content
    from tools.zoomrx_wiki_tool import handle_zoomrx_wiki_query
    logging.info("Successfully imported tool functions.")
except ImportError as e:
    logging.critical(f"DETAILED IMPORT ERROR: {str(e)}")
    logging.critical(f"Could not import tool functions. Current working directory: {os.getcwd()}")
    logging.critical(f"Python path: {os.sys.path}")
    logging.critical("Ensure tools/__init__.py exists and is properly imported")
    # Define dummy functions if imports fail, so the graph can still be defined (but won't work)
    def query_chroma_db(query: str): 
        logging.error("Dummy query_chroma_db called!")
        logging.error("Fix import errors to use the real implementation")
        return []
    def handle_panel_query(query: str, user_id: int): logging.error("Dummy handle_panel_query called!"); return "Panel tool not available."
    def get_conference_document_content(query: str): logging.error("Dummy get_conference_document_content called!"); return None
    def handle_zoomrx_wiki_query(query: str, user_id=None): logging.error("Dummy handle_zoomrx_wiki_query called!"); return "ZoomRx Wiki tool not available."

# --- Import Prompts ---
try:
    from .prompts import (
        CLASSIFICATION_SYSTEM_PROMPT,
        RESEARCH_RAG_PROMPT_TEMPLATE,
        PANEL_RAG_PROMPT_TEMPLATE,
        CONFERENCE_RAG_PROMPT_TEMPLATE,
        CONFERENCE_WEB_SEARCH_PROMPT_TEMPLATE,
        CONFERENCE_STRUCTURED_PROMPT,
        CONFERENCE_BULLETED_PROMPT,
        ZOOMRX_WIKI_PROMPT_TEMPLATE,
        RESEARCH_RAG_EXAMPLE_QUERY,
        RESEARCH_RAG_EXAMPLE_RESPONSE
    )
except ImportError:
    logging.critical("Could not import prompts. Ensure prompts.py exists in Agastya/agents/")
    # Define dummy prompts
    CLASSIFICATION_SYSTEM_PROMPT = "Classify query: 'panel_support', 'conference_info', 'research_lookup', 'greeting_chit_chat', 'unknown'."
    RESEARCH_RAG_PROMPT_TEMPLATE = "You are Medical-RAG-Bot, a clinical research assistant. Provide concise, well-structured medical research summaries from trusted sources."
    PANEL_RAG_PROMPT_TEMPLATE = "Panel Data: {context}\n\nUser: {userInput}\n\nAnswer based on panel data:"
    CONFERENCE_RAG_PROMPT_TEMPLATE = "Conference Info: {context}\n\nUser: {userInput}\n\nAnswer based on conference info:"
    CONFERENCE_WEB_SEARCH_PROMPT_TEMPLATE = "Please use web search for conference info. {context}\n\nUser: {userInput}\n\nAnswer:"
    CONFERENCE_STRUCTURED_PROMPT = "Please provide structured conference info. {context}\n\nUser: {userInput}\n\nAnswer in JSON format:"
    CONFERENCE_BULLETED_PROMPT = "Please provide bulleted conference info. {context}\n\nUser: {userInput}\n\nAnswer with bullets:"
    ZOOMRX_WIKI_PROMPT_TEMPLATE = "ZoomRx Wiki: {context}\n\nUser: {userInput}\n\nAnswer based on ZoomRx Wiki:"
    RESEARCH_RAG_EXAMPLE_QUERY = "What are the recent advances in pancreatic cancer treatment?"
    RESEARCH_RAG_EXAMPLE_RESPONSE = (
        "This summary synthesizes findings from PubMed and ClinicalTrials.gov on recent pancreatic cancer treatment advances.\n\n"
        
        "Key findings:\n"
        "- A phase II trial of FOLFIRINOX + nivolumab in 42 patients with locally advanced pancreatic cancer showed 38% partial response rate and median OS of 18.7 months, suggesting immunotherapy combinations may improve outcomes in selected patients.\n"
        "- Trial NCT05012345: An ongoing phase III study is testing mRNA-5671, a KRAS-targeted vaccine, in combination with pembrolizumab for metastatic disease after first-line chemotherapy (N=240).\n"
        "- A meta-analysis of 14 studies (N=1,062) found that neoadjuvant FOLFIRINOX for borderline resectable disease improved R0 resection rates by 28% compared to upfront surgery (p<0.001).\n\n"
        
        "Clinical implications:\n"
        "- Consider genetic testing for all pancreatic cancer patients to identify targetable mutations.\n"
        "- Neoadjuvant FOLFIRINOX continues to be preferred for borderline resectable disease.\n"
        "- Immune checkpoint inhibitors show promise for selected patients and in novel combinations.\n\n"
        
        "Limitations: This summary reflects data available as of May 10, 2024. Consult primary sources or specialists before making clinical decisions."
    )


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Agent State Definition ---
class AgentState(TypedDict):
    userInput: str
    userId: Optional[int] # Crucial for panel queries
    classifiedIntent: str
    previousIntent: Optional[str]  # Track the previous intent for follow-up detection
    conversationTopic: Optional[str]  # Track the current topic being discussed
    researchQuery: str # Potentially refined query for vector DB
    retrievedDocs: List[Dict] # List of dicts from query_chroma_db tool
    panelQueryResult: str
    conferenceQueryResult: Union[Dict, str] # Dict from tool, or error string
    zoomrxWikiResult: str # Result from ZoomRx Wiki tool
    finalAnswer: str
    chatHistory: Annotated[List[BaseMessage], operator.add]

# --- LLM and Retriever Initialization ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY not set in environment. LLM functionalities will fail.")
    # Provide dummy LLMs if key is missing to allow graph definition
    llm = None # type: ignore
    rag_llm = None # type: ignore
else:
    # Create LLM instances with model names from the configuration
    try:
        # Initialize model for intent classification
        llm = ChatOpenAI(
            model=get_model_for_task("intent_classification"), 
            openai_api_key=OPENAI_API_KEY, 
            temperature=1,
            model_kwargs={"max_completion_tokens": 150}
        ) 
        logger.info(f"Initialized intent classification LLM with model: {get_model_for_task('intent_classification')}")
        
        # Initialize models for different types of tasks - we'll create specific instances as needed
        logger.info(f"Using research synthesis model: {get_model_for_task('research_synthesis')}")
        logger.info(f"Using panel response model: {get_model_for_task('panel_response')}")
        logger.info(f"Using conference info model: {get_model_for_task('conference_info')}")
        logger.info(f"Using final response model: {get_model_for_task('final_response_generation')}")
    except Exception as e:
        logger.error(f"Error initializing LLM models: {e}", exc_info=True)
        llm = None
        rag_llm = None

# Placeholder for actual retriever initialization (would come from research_tool.py if it exposed the retriever object)
# For now, the research_tool.py handles its own retriever internally.

# --- Node Functions ---
def classify_intent_node_func(state: AgentState) -> Dict:
    logger.info("--- Node: Classifying Intent ---")
    userInput = state["userInput"]
    previousIntent = state.get("previousIntent")
    conversationTopic = state.get("conversationTopic")
    chatHistory = state.get("chatHistory", [])
    
    # Extract the previous AI message content for context if available
    previous_ai_message = None
    if chatHistory and len(chatHistory) >= 2:
        previous_ai_message = chatHistory[-1].content if isinstance(chatHistory[-1], AIMessage) else None
    
    logger.info(f"Previous intent: {previousIntent}, Conversation topic: {conversationTopic}")
    
    # Check if this is a follow-up question
    is_followup = False
    if previousIntent and userInput.lower().startswith(("tell me more", "what about", "and", "how about", "what is", "can you elaborate")):
        is_followup = True
        logger.info(f"Detected follow-up question pattern: {userInput}")
    
    # Topic continuation detection
    if previousIntent and previous_ai_message:
        # Extract key phrases from the previous response
        key_phrases = []
        # Split the previous response into sentences
        sentences = previous_ai_message.split('.')
        for sentence in sentences:
            # Extract phrases in quotes, numbered/bulleted items, or capitalized phrases
            if '"' in sentence or "'" in sentence:
                quoted = re.findall(r'["\'](.*?)["\']', sentence)
                key_phrases.extend(quoted)
            if re.search(r'^\s*[\d\*\-]+\.?\s+', sentence):
                key_phrases.append(sentence.strip())
            if "**" in sentence:
                bold = re.findall(r'\*\*(.*?)\*\*', sentence)
                key_phrases.extend(bold)
        
        # Check if current query contains any of these key phrases
        for phrase in key_phrases:
            if len(phrase) > 5 and phrase.lower() in userInput.lower():
                logger.info(f"Detected topic continuation with phrase: {phrase}")
                is_followup = True
                break
    
    # If this is a follow-up, maintain the previous intent
    if is_followup and previousIntent:
        logger.info(f"Maintaining previous intent {previousIntent} for follow-up question")
        return {
            "classifiedIntent": previousIntent,
            "researchQuery": userInput,
            "previousIntent": previousIntent,
            "conversationTopic": conversationTopic
        }
    
    # If not a follow-up, proceed with standard intent classification
    if not OPENAI_API_KEY: 
        return {"classifiedIntent": "unknown", "researchQuery": userInput, "finalAnswer": "LLM not initialized due to missing API key."}

    try:
        # Use the intent classification model
        intent_llm = ChatOpenAI(
            model=get_model_for_task("intent_classification"),
            openai_api_key=OPENAI_API_KEY,
            temperature=1,
            model_kwargs={"max_completion_tokens": 150}
        )
        
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", CLASSIFICATION_SYSTEM_PROMPT),
            ("human", "{userInput}")
        ])
        classification_chain = classification_prompt | intent_llm
        
        intent_result = classification_chain.invoke({"userInput": userInput})
        intent = intent_result.content.strip().lower().replace("'", "") # Remove quotes LLM might add
    except Exception as e:
        logger.error(f"Error during intent classification LLM call: {e}", exc_info=True)
        intent = "unknown" # Default to unknown on error

    # Extract topic from the query (simplified version)
    # A more sophisticated implementation would use NLP to extract entities
    topic_keywords = userInput.lower().split()
    topic = ' '.join([word for word in topic_keywords if len(word) > 3])[:50]  # Simple topic extraction

    # Basic keyword override/refinement (optional, can be improved)
    # First, check for ZoomRx product/offering questions - these should take precedence
    if (("zoomrx" in userInput.lower() or "zoom rx" in userInput.lower()) and 
        ("product" in userInput.lower() or "service" in userInput.lower() or 
         "offer" in userInput.lower() or "offering" in userInput.lower() or
         "outside of" in userInput.lower() or "besides" in userInput.lower() or
         "what is" in userInput.lower() or "what are" in userInput.lower() or
         "tell me about" in userInput.lower())):
        intent = "zoomrx_wiki"
        logger.info("Intent classification override: Routing to zoomrx_wiki due to product/offering query about ZoomRx")
    
    # FIRST CHECK: Personal earnings queries with explicit time indicators - these should go to panel_support
    elif (("earn" in userInput.lower() or "earned" in userInput.lower() or "earnings" in userInput.lower() or
           "money" in userInput.lower() or "payment" in userInput.lower() or "income" in userInput.lower()) and
          (("my" in userInput.lower() or "check" in userInput.lower() or "show" in userInput.lower() or 
            "view" in userInput.lower() or "i" in userInput.lower() or "how much" in userInput.lower()) and
           ("last" in userInput.lower() or "past" in userInput.lower() or "previous" in userInput.lower() or
            "month" in userInput.lower() or "year" in userInput.lower() or "week" in userInput.lower() or
            "recent" in userInput.lower() or "history" in userInput.lower()))):
        intent = "panel_support"
        logger.info("Intent classification override: Routing to panel_support due to personal earnings history query")
    
    # NEW CHECK: Personal earnings queries with specific year mention (like 2023, 2024, 2025) should go to panel_support
    elif (("earn" in userInput.lower() or "earned" in userInput.lower() or "earnings" in userInput.lower() or
           "money" in userInput.lower() or "payment" in userInput.lower() or "income" in userInput.lower()) and
          (("my" in userInput.lower() or "check" in userInput.lower() or "show" in userInput.lower() or 
            "view" in userInput.lower() or "i" in userInput.lower() or "how much" in userInput.lower()))):
        
        # Debug logging
        logger.info(f"Checking year pattern in: '{userInput.lower()}'")
        
        # More robust year pattern detection
        year_patterns = [
            r'(\d{4})',  # Basic 4-digit year
            r'in\s+(\d{4})',  # "in 2025"
            r'for\s+(\d{4})',  # "for 2025"
            r'during\s+(\d{4})',  # "during 2025"
        ]
        
        year_found = False
        for pattern in year_patterns:
            year_match = re.search(pattern, userInput.lower())
            if year_match:
                year_found = True
                logger.info(f"Year pattern match found: {year_match.group(1)}")
                intent = "panel_support"
                logger.info("Intent classification override: Routing to panel_support due to personal earnings with specific year query")
                break
        
        if not year_found:
            logger.info("No year pattern match found in earnings query")
    
    # Route general earnings questions about opportunities, potential earnings to zoomrx_wiki (check BEFORE panel_support)
    elif (("earn" in userInput.lower() or "earned" in userInput.lower() or "earnings" in userInput.lower() or
           "money" in userInput.lower() or "payment" in userInput.lower() or "income" in userInput.lower() or
           "compensation" in userInput.lower() or "paid" in userInput.lower() or "make money" in userInput.lower()) and
           ("increase" in userInput.lower() or "how" in userInput.lower() or "ways" in userInput.lower() or 
            "can" in userInput.lower() or "potential" in userInput.lower() or "opportunity" in userInput.lower() or 
            "opportunities" in userInput.lower() or "possible" in userInput.lower() or "more" in userInput.lower()) and
           not ("last" in userInput.lower() or "past" in userInput.lower() or "previous" in userInput.lower() or
                "month" in userInput.lower() or "year" in userInput.lower() or "week" in userInput.lower() or
                "recent" in userInput.lower() or "history" in userInput.lower())):
        intent = "zoomrx_wiki"
        logger.info("Intent classification override: Routing to zoomrx_wiki due to earnings opportunity query")
    
    # Then check for panel data queries about personal history/data
    elif (("panel" in userInput.lower() or "prescribing" in userInput.lower() or 
        "market share" in userInput.lower()) or
        # For surveys, only route to panel if it's about "my surveys" or "completed surveys" (personal data)
        (("surveys" in userInput.lower() or "participation" in userInput.lower()) and
         ("my" in userInput.lower() or "i" in userInput.lower() or 
          "complete" in userInput.lower() or "completed" in userInput.lower() or
          "took" in userInput.lower() or "taken" in userInput.lower() or 
          "did" in userInput.lower() or "finished" in userInput.lower() or
          "submitted" in userInput.lower())) or
        # Only route earnings-related questions to panel_support if they're specifically about personal earnings history:
        (("earn" in userInput.lower() or "earned" in userInput.lower() or "earnings" in userInput.lower() or
         "money" in userInput.lower() or "payment" in userInput.lower() or "income" in userInput.lower()) and
         # Check for personal context ("my", "I") AND check for time indicators
         (("my" in userInput.lower() or re.search(r'\bi\b', userInput.lower()) or "me" in userInput.lower() or "show" in userInput.lower() or "check" in userInput.lower()) and 
          ("history" in userInput.lower() or 
           "last" in userInput.lower() or "past" in userInput.lower() or 
           "recent" in userInput.lower() or "previous" in userInput.lower() or 
           "month" in userInput.lower() or "year" in userInput.lower() or 
           "week" in userInput.lower() or "day" in userInput.lower() or
           "january" in userInput.lower() or "february" in userInput.lower() or
           "march" in userInput.lower() or "april" in userInput.lower() or
           "may" in userInput.lower() or "june" in userInput.lower() or
           "july" in userInput.lower() or "august" in userInput.lower() or
           "september" in userInput.lower() or "october" in userInput.lower() or
           "november" in userInput.lower() or "december" in userInput.lower() or
           "q1" in userInput.lower() or "q2" in userInput.lower() or 
           "q3" in userInput.lower() or "q4" in userInput.lower() or
           "so far" in userInput.lower() or "to date" in userInput.lower() or
           "have earned" in userInput.lower() or "earned so far" in userInput.lower() or 
           "have i" in userInput.lower() or "did i" in userInput.lower())))):
        intent = "panel_support"
        logger.info("Intent classification override: Routing to panel_support due to personal data query")
    
    # Other types of ZoomRx questions - expanded to capture more keywords
    elif ("zoomrx" in userInput.lower() or "zoom rx" in userInput.lower() or 
          "product" in userInput.lower() or "service" in userInput.lower() or 
          "offer" in userInput.lower() or "hcp-pt" in userInput.lower() or 
          "hcp-patient" in userInput.lower() or "advisory board" in userInput.lower() or
          "perxcept" in userInput.lower() or "digital tracker" in userInput.lower() or "web extension" in userInput.lower() or
          "referral" in userInput.lower() or "refer" in userInput.lower() or
          "referrals" in userInput.lower() or "referral program" in userInput.lower() or
          "referring" in userInput.lower() or "browser extension" in userInput.lower() or
          "record conversations" in userInput.lower() or "dialogue" in userInput.lower() or
          "patient recording" in userInput.lower() or "recording patients" in userInput.lower() or
          "forward emails" in userInput.lower() or "email forwarding" in userInput.lower() or
          "hcp survey" in userInput.lower() or "survey offering" in userInput.lower() or
          "survey offerings" in userInput.lower() or "types of survey" in userInput.lower() or
          "survey types" in userInput.lower() or "types of surveys" in userInput.lower()):
        intent = "zoomrx_wiki"
        logger.info("Intent classification override: Routing to zoomrx_wiki due to ZoomRx mention or feature query")
    
    # Conference queries
    elif "conference" in userInput.lower() or "asco" in userInput.lower() or "esmo" in userInput.lower() or "acc" in userInput.lower():
        intent = "conference_info"
        logger.info("Intent classification override: Routing to conference_info due to conference keyword")
    elif intent not in ["panel_support", "conference_info", "research_lookup", "greeting_chit_chat", "zoomrx_wiki"]:
        intent = "unknown"
        
    logger.info(f"Classified Intent: {intent} for input: '{userInput}'")
    return {
        "classifiedIntent": intent, 
        "researchQuery": userInput,
        "previousIntent": intent,  # Store the current intent as previous for next query
        "conversationTopic": topic  # Store the topic for context maintenance
    }

def research_lookup_node_func(state: AgentState) -> Dict:
    logger.info("--- Node: Performing Research Lookup ---")
    query = state.get("researchQuery", state["userInput"]) # Use refined query if available
    
    retrieved_data = query_chroma_db(query) # Call tool from tools/research_tool.py
    
    logger.info(f"Research Lookup: Retrieved {len(retrieved_data)} documents for query '{query}'.")
    return {"retrievedDocs": retrieved_data}

def panel_support_node_func(state: AgentState) -> Dict:
    logger.info("--- Node: Performing Panel Support Query ---")
    userInput = state["userInput"]
    userId = state.get("userId")

    if userId is None:
        logger.error("Panel Support: User ID not found in state!")
        return {"panelQueryResult": "Error: I need a user ID to access panel data. This should be set during login."}
        
    result = handle_panel_query(userInput, userId) # Call tool from tools/panel_tool.py
    
    logger.info(f"Panel Support: Result for user {userId}, query '{userInput}': '{str(result)[:200]}...'")
    return {"panelQueryResult": result}

def conference_info_node_func(state: AgentState) -> Dict:
    logger.info("--- Node: Performing Conference Info Lookup ---")
    userInput = state["userInput"]
    
    conference_data_dict = get_conference_document_content(userInput) # Call tool from tools/conference_tool.py
    
    # Prepare context string for the RAG LLM
    if conference_data_dict:
        # Check if this is a web search result from Tavily
        if conference_data_dict.get("source") == "web_search":
            logger.info(f"Conference Info: Processing Tavily search results for query '{userInput}'")
            
            # If search was successful, return the results directly
            if conference_data_dict.get("success", False):
                return {"conferenceQueryResult": {
                    "type": "tavily_search",
                    "success": True,
                    "search_query": conference_data_dict.get("search_query", ""),
                    "description": conference_data_dict.get("description", ""),
                    "content": conference_data_dict.get("content", ""),
                    "original_query": userInput
                }}
            else:
                # If Tavily search failed, return the error
                error_message = conference_data_dict.get("content", "Unknown error with web search")
                logger.warning(f"Conference Info: Tavily search failed: {error_message}")
                return {"conferenceQueryResult": {
                    "type": "tavily_search_error",
                    "success": False,
                    "error": error_message,
                    "original_query": userInput
                }}
        
        # Handle PDF content as before
        elif conference_data_dict.get("content"):
            if "Error: Could not extract content" in conference_data_dict["content"]:
                prepared_context = conference_data_dict["content"] # Pass error message as context
            else:
                prepared_context = (f"Regarding the conference document '{conference_data_dict['pdf_name']}' "
                                   f"(which is about {conference_data_dict['description']}):\n\n"
                                   f"{conference_data_dict['content']}")
            logger.info(f"Conference Info: Prepared context from '{conference_data_dict.get('pdf_name', 'Unknown PDF')}'.")
            return {"conferenceQueryResult": prepared_context}
    else:
        prepared_context = "I could not find specific conference document information related to your query."
        logger.warning(f"Conference Info: No relevant document or content found for query '{userInput}'.")
        
    return {"conferenceQueryResult": prepared_context}

def zoomrx_wiki_node_func(state: AgentState) -> Dict:
    logger.info("--- Node: Performing ZoomRx Wiki Lookup ---")
    userInput = state["userInput"]
    userId = state.get("userId")
    
    try:
        result = handle_zoomrx_wiki_query(userInput, userId) # Call tool from tools/zoomrx_wiki_tool.py
        logger.info(f"ZoomRx Wiki: Result for query '{userInput}': '{str(result)[:200]}...'")
        return {"zoomrxWikiResult": result}
    except Exception as e:
        error_message = f"Error processing ZoomRx Wiki query: {str(e)}"
        logger.error(error_message, exc_info=True)
        return {"zoomrxWikiResult": error_message}

def generate_response_node_func(state: AgentState) -> Dict:
    logger.info("--- Node: Generating Final Response ---")
    userInput = state["userInput"]
    intent = state["classifiedIntent"]
    previousIntent = state.get("previousIntent")
    conversationTopic = state.get("conversationTopic")
    current_history = state.get("chatHistory", [])
    final_answer = "Sorry, I encountered an issue generating a response." # Default

    if not OPENAI_API_KEY: 
        return {"finalAnswer": "RAG LLM not initialized due to missing API key.", 
                "chatHistory": current_history + [HumanMessage(content=userInput), AIMessage(content=final_answer)],
                "previousIntent": intent,
                "conversationTopic": conversationTopic}

    context_for_llm = ""
    prompt_template_for_rag = None
    model_task = "default"  # Default task identifier

    if intent == "research_lookup":
        retrieved_data = state.get("retrievedDocs", [])
        if retrieved_data:
            # Better formatting of research documents for inclusion in the prompt
            context_parts = []
            for i, doc_data in enumerate(retrieved_data[:10]):  # Limit to first 10 documents to avoid context overflows
                metadata = doc_data.get('metadata', {})
                title = metadata.get('original_doc_title', f'Document {i+1}')
                source = metadata.get('original_doc_source', 'Unknown source')
                pub_date = metadata.get('original_doc_publication_date', 'Unknown date')
                url = metadata.get('original_doc_url', '#')
                # Format the document with clear section headers
                doc_context = (
                    f"SOURCE {i+1}:\n"
                    f"Title: {title}\n"
                    f"Source: {source}\n"
                    f"Publication Date: {pub_date}\n"
                    f"URL: {url}\n"
                    f"Content: {doc_data['page_content']}\n"
                )
                context_parts.append(doc_context)
            
            context_for_llm = "\n\n---\n\n".join(context_parts)
            logger.info(f"Research RAG: Prepared context from {len(retrieved_data[:10])} documents")
        else:
            context_for_llm = "No relevant documents were found in the knowledge base for your research query."
            logger.warning("Research RAG: No documents retrieved")
        
        try:
            # Create a task-specific LLM instance for research synthesis
            from datetime import datetime
            current_date = datetime.now().strftime("%B %d, %Y")
            
            task_llm = ChatOpenAI(
                model=get_model_for_task("research_synthesis"),
                openai_api_key=OPENAI_API_KEY,
                temperature=0.7,
                model_kwargs={"max_completion_tokens": 1000}
            )
            
            # Use RESEARCH_RAG_PROMPT_TEMPLATE as system message and include example
            messages = [
                {"role": "system", "content": RESEARCH_RAG_PROMPT_TEMPLATE},
                {"role": "user", "content": RESEARCH_RAG_EXAMPLE_QUERY},
                {"role": "assistant", "content": RESEARCH_RAG_EXAMPLE_RESPONSE},
                {"role": "user", "content": f"{userInput}\n\nRetrieved Context:\n{context_for_llm}"}
            ]
            
            # Process messages through the OpenAI API
            response = task_llm.invoke(messages)
            
            # Replace any static dates with the current date in the limitations section
            final_answer = response.content
            if "Limitations: This summary reflects data available as of" in final_answer:
                final_answer = final_answer.replace("May 10, 2024", current_date)
                final_answer = final_answer.replace("{{today}}", current_date)
                final_answer = final_answer.replace("{today}", current_date)
            elif "Limitations:" not in final_answer:
                final_answer += f"\n\nLimitations: This summary reflects data available as of {current_date}. Consult primary sources or specialists before making clinical decisions."
            
            logger.info(f"Research RAG: Successfully generated research synthesis response")
        except Exception as e:
            logger.error(f"Error during Research RAG LLM call: {e}", exc_info=True)
            final_answer = f"Sorry, I encountered an issue while synthesizing research information for your query. Please try again or rephrase your question."
    
    elif intent == "panel_support":
        context_for_llm = state.get("panelQueryResult", "No panel data was retrieved.")
        prompt_template_for_rag = PANEL_RAG_PROMPT_TEMPLATE
        model_task = "panel_response"
        
    elif intent == "conference_info":
        conference_result = state.get("conferenceQueryResult", "No conference information was retrieved.")
        
        # Check if this is a Tavily search result
        if isinstance(conference_result, dict) and conference_result.get("source") == "web_search":
            # This is a successful Tavily search
            search_info = conference_result
            
            # Check if structured output is needed
            if search_info.get("success", False):
                # Default to bulleted format for visual appeal, only use JSON if explicitly requested
                use_json_format = "json" in userInput.lower() or "structured json" in userInput.lower()
                
                if use_json_format:
                    # Use structured JSON prompt if JSON specifically requested
                    context_for_llm = (
                        f"The following are web search results related to the conference.\n\n"
                        f"{search_info.get('content', '')}\n\n"
                        f"Provide comprehensive structured information about this conference based on the search results."
                    )
                    logger.info(f"Conference Info: Using Tavily search results with JSON structured output format")
                    prompt_template_for_rag = CONFERENCE_STRUCTURED_PROMPT
                else:
                    # Use bulleted format for better visual presentation (default)
                    context_for_llm = (
                        f"The following are web search results related to the conference.\n\n"
                        f"{search_info.get('content', '')}\n\n"
                        f"Provide comprehensive bulleted information about this conference based on the search results."
                    )
                    logger.info(f"Conference Info: Using Tavily search results with bulleted format")
                    prompt_template_for_rag = CONFERENCE_BULLETED_PROMPT
            else:
                # Error handling for failed searches
                context_for_llm = (
                    f"I'm sorry, but the search for conference information was not successful. "
                    f"Search Query: {search_info.get('search_query', 'Unknown')}\n"
                    f"Error: {search_info.get('content', 'Unknown error')}\n\n"
                    f"Please try again with a more specific query or check if the Tavily API is properly configured."
                )
                logger.info(f"Conference Info: Tavily search failed")
                prompt_template_for_rag = CONFERENCE_RAG_PROMPT_TEMPLATE
        
        # Check if this is a web search recommendation (legacy code, kept for compatibility)
        elif isinstance(conference_result, dict) and conference_result.get("type") == "web_search":
            web_search_info = conference_result
            
            context_for_llm = (
                f"I apologize, but I don't have specific up-to-date information about this conference. "
                f"For the most current details about {web_search_info.get('search_query', 'this medical conference')}, "
                f"I recommend visiting the official conference website or searching for '{web_search_info.get('search_query', 'medical conference information')}'."
            )
            
            logger.info(f"Conference Info: Using legacy web search approach (not functional)")
            prompt_template_for_rag = CONFERENCE_RAG_PROMPT_TEMPLATE
        else:
            # Regular PDF-based context
            context_for_llm = conference_result
            prompt_template_for_rag = CONFERENCE_RAG_PROMPT_TEMPLATE
            
        model_task = "conference_info"
        
    elif intent == "zoomrx_wiki":
        context_for_llm = state.get("zoomrxWikiResult", "")
        prompt_template_for_rag = ZOOMRX_WIKI_PROMPT_TEMPLATE
        model_task = "zoomrx_wiki_response"
        
    elif intent == "greeting_chit_chat":
        try:
            # Try to get user's name
            try:
                from tools.user_info import get_user_name
                userId = state.get("userId")
                first_name, last_name = get_user_name(userId)
                doctor_name = f"Dr. {last_name}" if last_name else f"User {userId}"
            except Exception as e:
                logger.error(f"Error getting user name: {e}")
                doctor_name = f"User {userId}"
                
            # Use the default model for simple responses
            chat_llm = ChatOpenAI(
                model=get_model_for_task("default"),
                openai_api_key=OPENAI_API_KEY,
                temperature=1.0,
                model_kwargs={"max_completion_tokens": 100}
            )
            
            # Use a simpler prompt or direct logic for greetings
            if "hello" in userInput.lower() or "hi" in userInput.lower():
                final_answer = f"Hello {doctor_name}! How can I help you today with medical research, panel data, or conference information?"
            elif "thank you" in userInput.lower() or "thanks" in userInput.lower():
                final_answer = f"You're welcome, {doctor_name}! Is there anything else I can assist with?"
            else:
                final_answer = f"I can help with medical research, panel data, or conference information. What would you like to know, {doctor_name}?"
            updated_history = current_history + [HumanMessage(content=userInput), AIMessage(content=final_answer)]
            return {"finalAnswer": final_answer, "chatHistory": updated_history, "previousIntent": intent, "conversationTopic": conversationTopic}
        except Exception as e:
            logger.error(f"Error handling greeting: {e}", exc_info=True)
            final_answer = "Hello! How can I assist you today?"
            updated_history = current_history + [HumanMessage(content=userInput), AIMessage(content=final_answer)]
            return {"finalAnswer": final_answer, "chatHistory": updated_history, "previousIntent": intent, "conversationTopic": conversationTopic}
    
    else: # Should be caught by handle_unknown if intent was 'unknown'
        final_answer = "I'm not equipped to handle that type of request."
        updated_history = current_history + [HumanMessage(content=userInput), AIMessage(content=final_answer)]
        return {"finalAnswer": final_answer, "chatHistory": updated_history, "previousIntent": intent, "conversationTopic": conversationTopic}

    # Process non-research intents that use the standard template approach
    if intent != "research_lookup" and prompt_template_for_rag:
        try:
            # Create a task-specific LLM instance based on the intent
            task_llm = ChatOpenAI(
                model=get_model_for_task(model_task),
                openai_api_key=OPENAI_API_KEY,
                temperature=1.0,
                model_kwargs={"max_completion_tokens": 1000}
            )
            
            generation_prompt = ChatPromptTemplate.from_template(prompt_template_for_rag)
            generation_chain = generation_prompt | task_llm
            final_answer = generation_chain.invoke({"context": context_for_llm, "userInput": userInput}).content
            
            # For the final response, we might want to use a different model than what was used for synthesis
            if model_task != "final_response_generation":
                logger.info(f"Using task-specific model for {model_task}, not applying final response model")
        except Exception as e:
            logger.error(f"Error during RAG LLM call for intent '{intent}': {e}", exc_info=True)
            final_answer = f"Sorry, I encountered an issue while generating the response for your {intent.replace('_', ' ')} query."
    
    updated_history = current_history + [HumanMessage(content=userInput), AIMessage(content=final_answer)]
    logger.info(f"Final generated answer (snippet): {final_answer[:100]}...")
    return {"finalAnswer": final_answer, "chatHistory": updated_history, "previousIntent": intent, "conversationTopic": conversationTopic}

def handle_unknown_node_func(state: AgentState) -> Dict:
    logger.info("--- Node: Handling Unknown Intent ---")
    answer = "I'm not sure how to help with that. I can assist with queries about medical research, physician panel data, or conference information. Could you please rephrase?"
    userInput = state["userInput"]
    current_history = state.get("chatHistory", [])
    intent = state.get("classifiedIntent", "unknown")
    conversationTopic = state.get("conversationTopic")
    updated_history = current_history + [HumanMessage(content=userInput), AIMessage(content=answer)]
    return {
        "finalAnswer": answer, 
        "chatHistory": updated_history, 
        "previousIntent": intent,
        "conversationTopic": conversationTopic
    }


# --- Conditional Edge Router ---
def route_after_classification(state: AgentState) -> str:
    intent = state["classifiedIntent"]
    logger.info(f"Routing based on intent: {intent}")
    if intent == "research_lookup":      return "research_tool"
    elif intent == "panel_support":      return "panel_tool"
    elif intent == "conference_info":    return "conference_tool"
    elif intent == "zoomrx_wiki":        return "zoomrx_wiki_tool"
    elif intent == "greeting_chit_chat": return "generate_final_response" # Directly generate simple response
    else: # unknown or other
                                         return "handle_unknown_intent"

# --- Build the LangGraph Workflow ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify_intent", classify_intent_node_func)
workflow.add_node("research_tool", research_lookup_node_func)
workflow.add_node("panel_tool", panel_support_node_func)
workflow.add_node("conference_tool", conference_info_node_func)
workflow.add_node("zoomrx_wiki_tool", zoomrx_wiki_node_func)
workflow.add_node("generate_final_response", generate_response_node_func)
workflow.add_node("handle_unknown_intent", handle_unknown_node_func)

# Set entry point
workflow.set_entry_point("classify_intent")

# Add conditional edges from classification
workflow.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "research_tool": "research_tool",
        "panel_tool": "panel_tool",
        "conference_tool": "conference_tool",
        "zoomrx_wiki_tool": "zoomrx_wiki_tool",
        "generate_final_response": "generate_final_response", # For greeting_chit_chat
        "handle_unknown_intent": "handle_unknown_intent"
    }
)

# Add edges from tools to response generation
workflow.add_edge("research_tool", "generate_final_response")
workflow.add_edge("panel_tool", "generate_final_response")
workflow.add_edge("conference_tool", "generate_final_response")
workflow.add_edge("zoomrx_wiki_tool", "generate_final_response")

# Add end points
workflow.add_edge("handle_unknown_intent", END)
workflow.add_edge("generate_final_response", END)

# Compile the graph without checkpointing
# Note: We're not using SQLite checkpointing as it requires thread_id/thread_ts configuration
app = workflow.compile()
logger.info("LangGraph workflow compiled without checkpointing.")

logger.info("LangGraph workflow compiled.")

# To run this graph, you would typically do it from your main run.py or main_agent.py:
# if __name__ == '__main__':
#     # Example run (ensure OPENAI_API_KEY is set in .env or environment)
#     if not OPENAI_API_KEY:
#          print("CRITICAL: OPENAI_API_KEY must be set to run this example.")
#     else:
#         print("Starting LangGraph flow example. Type 'exit' to quit.")
#         current_chat_history = []
#         example_user_id = 12345 # Example user ID for panel queries

#         while True:
#             user_q = input("You: ")
#             if user_q.lower() == 'exit':
#                 break
            
#             inputs = {"userInput": user_q, "userId": example_user_id, "chatHistory": current_chat_history}
#             final_state_after_run = None
            
#             print("Agastya is thinking...")
#             # Using stream to see intermediate states, invoke for just final result
#             for event_chunk in app.stream(inputs, stream_mode="values"):
#                 # print(f"\n--- Agent State Update ---")
#                 # for key, value_item in event_chunk.items():
#                 #     print(f"  {key}: {str(value_item)[:300]}") # Print snippet
#                 final_state_after_run = event_chunk 
            
#             if final_state_after_run and final_state_after_run.get("finalAnswer"):
#                 ai_resp = final_state_after_run["finalAnswer"]
#                 print(f"Agastya: {ai_resp}")
#                 current_chat_history = final_state_after_run.get("chatHistory", [])
#                 # Truncate history if it gets too long
#                 if len(current_chat_history) > 10: # Keep last 5 pairs
#                     current_chat_history = current_chat_history[-10:]
#             else:
#                 print("Agastya: Sorry, I couldn't process that.")
#                 current_chat_history.append(HumanMessage(content=user_q))
#                 current_chat_history.append(AIMessage(content="I had an issue."))