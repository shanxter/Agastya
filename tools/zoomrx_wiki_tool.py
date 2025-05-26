import os
import json
import logging
import re
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the knowledge base
WIKI_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zoomrx_wiki_data.json')
WIKI_DATA = {}

try:
    with open(WIKI_DATA_PATH, 'r') as f:
        WIKI_DATA = json.load(f)
    logger.info(f"Loaded ZoomRx Wiki data from {WIKI_DATA_PATH}")
except Exception as e:
    logger.error(f"Error loading ZoomRx Wiki data: {e}")

# Track conversation context (simple in-memory store - could be replaced with a more persistent solution)
# Format: {user_id: {"last_product": product_name, "last_query": query, "follow_up_offered": bool}}
CONVERSATION_CONTEXT = {}

def similarity_score(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_best_match(query, candidates):
    """Find the best matching item from candidates based on similarity to query."""
    best_score = 0
    best_match = None
    
    for candidate in candidates:
        score = similarity_score(query, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
    
    # Return the match only if it's reasonably good
    if best_score > 0.6:
        return best_match
    return None

def extract_product_mentions(query):
    """Extract mentions of products from the query."""
    query_lower = query.lower()
    
    # Simple keyword matching for products
    product_keywords = {
        "hcp_surveys": [
            "survey", "surveys", "hcp survey", "medical survey", "healthcare survey", 
            "questionnaire", "physician survey", "doctor survey", "hcp surveys", 
            "medical questionnaire", "polling", "survey offering", "survey offerings",
            "types of survey", "survey types", "types of surveys", "chart review",
            "patient chart", "chart submission", "sales rep interaction", "qualitative interview",
            "interview", "chart reviews", "traditional survey"
        ],
        "hcp_pt": [
            "patient", "patient connect", "hcp-pt", "hcp pt", "hcp-patient", 
            "dialogue", "dialog", "conversation", "recording", "audio", 
            "hcp-patient dialogue", "patient dialogue", "patient dialog", 
            "clinical conversation", "doctor-patient", "physician-patient",
            "hcp patient", "patient dialogue", "patient recordings", "clinical recordings",
            "record conversations", "record patient", "record meeting", "patient recording", 
            "conversation recording", "record visit", "record consultation",
            "conversation feature", "recording feature", "audio recording"
        ],
        "advisory_boards": [
            "advisory", "board", "advisory board", "virtual advisory", "expert panel", 
            "consulting", "advisor", "expert input", "advisory boards", 
            "pharmaceutical advisory", "medical advisory", "consulting opportunity"
        ],
        "digital_tracker": [
            "digital tracker", "perxcept", "digital insights", "passive", "browser extension", "safari extension",
            "email forwarding", "web tracking", "healthcare content tracking", "online behavior",
            "healthcare emails", "digital monitoring", "browser plugin", "web extension",
            "extension", "plugin", "monitoring tool", "content tracker", "browser monitor",
            "web monitoring", "passive monitoring", "passive tracking", "safari plugin",
            "email forwarding feature", "forward emails", "email tracking"
        ]
    }
    
    # Handle more generic feature-based queries
    generic_feature_mapping = {
        "web extension": "digital_tracker",
        "browser extension": "digital_tracker",
        "browser plugin": "digital_tracker",
        "record conversations": "hcp_pt",
        "record patients": "hcp_pt",
        "patient recordings": "hcp_pt",
        "conversation recording": "hcp_pt",
        "forward emails": "digital_tracker",
        "track online": "digital_tracker",
        "passive monitoring": "digital_tracker",
        "digital tracker": "digital_tracker"
    }
    
    # Check for generic feature descriptions first
    for feature_desc, product_id in generic_feature_mapping.items():
        if feature_desc in query_lower:
            return [product_id]
    
    # Check for referral program mentions
    if any(term in query_lower for term in ["referral", "refer", "referring", "referrals", "referral program", "refer a colleague", "refer a friend"]):
        return ["referral_program"]
    
    # Don't extract product names in comparison queries
    if any(phrase in query_lower for phrase in ["outside of", "besides", "other than", "apart from", 
                                               "in addition to", "what else", "what other"]):
        print("Detected comparison query - not extracting products directly")
        return []
    
    mentioned_products = []
    
    for product_id, keywords in product_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                # To avoid double-matching
                if product_id not in mentioned_products:
                    mentioned_products.append(product_id)
                break
    
    return mentioned_products

def identify_question_type(query):
    """Identify what type of question the user is asking."""
    query_lower = query.lower()
    
    # Enhanced pattern matching for different question types with more conversational variants
    
    # Check for feature-related queries
    if any(term in query_lower for term in ["feature", "features", "functionality", "capabilities", 
                                           "tools", "what can", "functions", "options", 
                                           "what does it do", "what can it do", "how does it work",
                                           "web extension", "browser extension", "record conversation",
                                           "record patients", "recordings"]):
        return "features"
    
    # Check for product-related queries (highest priority for offering questions)
    if any(phrase in query_lower for phrase in [
        "outside of", "besides", "other than", "apart from", "in addition to",
        "what else", "what other", "different from", "different than",
        "other products", "other offerings", "other services"
    ]):
        return "products"
    
    if any(term in query_lower for phrase in ["offer", "offers", "offering", "offerings", 
                                            "product", "products", "service", "services"] 
                            for term in [phrase + "s", phrase]):
        return "products"
    
    if any(term in query_lower for term in ["what is", "what are", "tell me about", "describe", "explain",
                                           "what does", "who is", "information on", "info about"]):
        return "what_is"
    elif any(term in query_lower for term in ["benefit", "advantage", "value", "why", "purpose",
                                             "why should i", "what's in it for me", "what's good about",
                                             "reasons to", "pros of"]):
        return "benefits"
    elif any(term in query_lower for term in ["how to", "how do i", "start", "join", "sign up", "enroll",
                                             "participate", "register", "begin", "get going", "onboarding"]):
        return "how_to"
    elif any(phrase in query_lower for phrase in ["earn", "payment", "money", "compensation", "pay", "income", "salary",
                                                 "how much", "make money", "get paid", "earnings", "payout",
                                                 "earn more", "earn extra", "additional income", "side income"]):
        return "earnings"
    elif any(term in query_lower for term in ["qualification", "qualify", "eligible", "eligibility",
                                             "who can", "requirements", "can i", "am i eligible",
                                             "do i qualify", "prerequisites"]):
        return "eligibility"
    elif any(term in query_lower for term in ["faq", "question", "common question", "ask", "frequently",
                                             "people ask", "common concern"]):
        return "faqs"
    elif any(term in query_lower for term in ["product", "products", "offering", "offerings", "services",
                                             "programs", "options", "opportunities", "activities",
                                             "ways to participate"]):
        return "products"
    elif any(term in query_lower for term in ["time", "commitment", "hours", "schedule", "availability",
                                             "how long", "time required", "time commitment"]):
        return "time_commitment"
    else:
        return "general"

def format_product_info(product_id, section=None):
    """Format product information for a response."""
    # Special handling for referral program
    if product_id == "referral_program":
        referral_info = WIKI_DATA["general_info"]["earning_opportunities"]["referral_program"]
        
        response = "# ZoomRx Referral Program\n\n"
        response += "The ZoomRx referral program offers you additional earning opportunities by inviting your colleagues and patients to join.\n\n"
        
        response += "## HCP Referrals\n"
        response += f"{referral_info['hcp_referral']}\n\n"
        
        response += "## Patient Referrals\n"
        response += f"{referral_info['patient_referral']}\n\n"
        
        response += "To get your referral link, log in to your ZoomRx account, navigate to the top of the dashboard, and select 'Refer & Earn'."
        
        return response
    
    # Regular product handling
    if product_id not in WIKI_DATA["products"]:
        return "I don't have information about that specific product."
    
    product = WIKI_DATA["products"][product_id]
    
    # Provide context translation for generic feature queries
    feature_context = ""
    if product_id == "digital_tracker" and section in ["what_is", None]:
        feature_context = "This is the ZoomRx **Digital Tracker** product, which provides the web extension/browser monitoring functionality.\n\n"
    elif product_id == "hcp_pt" and section in ["what_is", None]:
        feature_context = "This is the ZoomRx **HCP-Patient Dialogue** product, which provides the patient conversation recording functionality.\n\n"
    
    if section == "what_is":
        return f"{feature_context}{product['name']}: {product['description']}"
    elif section == "how_to":
        return f"{feature_context}Getting started with {product['name']}:\n{product['how_to_start']}"
    elif section == "earnings":
        return f"{feature_context}Earnings for {product['name']}:\n{product['earnings']}"
    elif section == "benefits":
        benefits_text = "\n".join([f"• {b}" for b in product['benefits']])
        return f"{feature_context}Benefits of {product['name']}:\n{benefits_text}"
    elif section == "faqs":
        faqs_text = "\n\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in product.get('faqs', [])])
        if faqs_text:
            return f"{feature_context}Frequently Asked Questions about {product['name']}:\n\n{faqs_text}"
        return f"{feature_context}I don't have specific FAQs about {product['name']}."
    else:
        # Comprehensive overview
        response = f"# {product['name']}\n\n"
        response += feature_context
        response += f"{product['description']}\n\n"
        
        response += "## Benefits\n"
        response += "\n".join([f"• {b}" for b in product['benefits']])
        response += f"\n\n## How to Get Started\n{product['how_to_start']}"
        response += f"\n\n## Earnings\n{product['earnings']}"
        
        if product.get('faqs'):
            response += "\n\n## Frequently Asked Questions\n"
            for faq in product['faqs']:
                response += f"\nQ: {faq['question']}\nA: {faq['answer']}\n"
        
        return response

def get_general_info(section=None):
    """Get general information about ZoomRx."""
    if section == "about" or section == "what_is":
        return WIKI_DATA["general_info"]["about_zoomrx"]
    elif section == "earnings":
        # Enhanced earnings response with specific product details
        response = f"## Earnings Potential with ZoomRx\n\n{WIKI_DATA['general_info']['earnings_potential']}\n\n"
        response += "### Specific Earning Opportunities:\n\n"
        
        for product_id, product in WIKI_DATA["products"].items():
            response += f"**{product['name']}**: {product['earnings']}\n\n"
            
        response += "Would you like more details about any of these specific earning opportunities?"
        return response
    elif section == "eligibility":
        return f"Eligibility for ZoomRx:\n{WIKI_DATA['participation_info']['eligibility']}"
    elif section == "time_commitment":
        return f"Time Commitment:\n{WIKI_DATA['participation_info']['time_commitment']}"
    elif section == "payment_methods":
        return f"Payment Methods:\n{WIKI_DATA['participation_info']['payment_methods']}"
    else:
        # General overview
        response = "# About ZoomRx\n\n"
        response += WIKI_DATA["general_info"]["about_zoomrx"] + "\n\n"
        response += f"## Our Mission\n{WIKI_DATA['general_info']['company_mission']}\n\n"
        response += f"## Earnings Potential\n{WIKI_DATA['general_info']['earnings_potential']}\n\n"
        response += f"## Data Privacy\n{WIKI_DATA['general_info']['data_privacy']}\n\n"
        response += f"## Eligibility\n{WIKI_DATA['participation_info']['eligibility']}\n\n"
        response += f"## Time Commitment\n{WIKI_DATA['participation_info']['time_commitment']}\n\n"
        response += f"## Payment Methods\n{WIKI_DATA['participation_info']['payment_methods']}"
        return response

def generate_follow_up(product_id=None, question_type=None):
    """Generate a relevant follow-up question based on the context."""
    # Product-specific follow-ups
    if product_id:
        if question_type == "earnings":
            return f"Would you like to learn more about how to get started with {WIKI_DATA['products'][product_id]['name']} to begin earning?"
        elif question_type == "benefits":
            return f"Would you like to learn how much you could earn with {WIKI_DATA['products'][product_id]['name']}?"
        elif question_type == "features":
            if product_id == "digital_tracker":
                return f"Would you like to learn how to set up the {WIKI_DATA['products'][product_id]['name']} web extension on your browser?"
            elif product_id == "hcp_pt":
                return f"Would you like to learn how to start recording patient conversations with {WIKI_DATA['products'][product_id]['name']}?"
            else:
                return f"Would you like to learn more about how to get started with {WIKI_DATA['products'][product_id]['name']}?"
        else:
            return f"Would you like to learn how to get started with {WIKI_DATA['products'][product_id]['name']}?"
    # General follow-ups
    else:
        if question_type == "earnings":
            return "Would you like to know more about any of these specific earning opportunities?"
        elif question_type == "eligibility":
            return "Would you like to know which ZoomRx opportunity might be the best fit for your specialty?"
        elif question_type == "features":
            return "Which of these features would you like to learn more about: web extension (Digital Tracker), patient recordings (HCP-Patient Dialogue), or surveys?"
        else:
            return "Would you like to know more about any specific ZoomRx product or service?"

def handle_zoomrx_wiki_query(user_query, user_id=None):
    """
    Main function to handle wiki queries about ZoomRx.
    
    Args:
        user_query: The user's question about ZoomRx
        user_id: Optional user ID for tracking conversation context
    
    Returns:
        A response to the user's query with appropriate information
    """
    if not WIKI_DATA:
        return "I'm sorry, but the ZoomRx knowledge base is not available. Please try again later."
    
    # Parse the query to understand what's being asked
    question_type = identify_question_type(user_query)
    query_lower = user_query.lower()
    print(f"Query: '{user_query}', Identified question type: {question_type}")
    
    # Special handling for comparison queries (outside of X, besides Y)
    comparison_keywords = ["outside of", "besides", "other than", "apart from", 
                          "in addition to", "what else", "what other"]
    is_comparison_query = any(keyword in query_lower for keyword in comparison_keywords)
    
    # Force products question type for comparison queries
    if is_comparison_query:
        question_type = "products"
        print(f"Detected comparison query. Forcing question_type to 'products'")
    
    # Detect feature-based queries
    feature_based_query = any(term in query_lower for term in [
        "web extension", "browser extension", "record conversations", "record patients",
        "feature", "functionality", "tool", "capability", "function"
    ])
    print(f"Feature-based query: {feature_based_query}")
    
    mentioned_products = extract_product_mentions(user_query)
    print(f"Mentioned products: {mentioned_products}")
    
    # Special case for when they ask about available features
    if question_type == "features" and not mentioned_products:
        response = "# ZoomRx Features\n\n"
        response += "ZoomRx offers several key features for healthcare professionals:\n\n"
        
        response += "## Web Extension (Digital Tracker)\n"
        response += f"{WIKI_DATA['products']['digital_tracker']['description']}\n"
        response += "Earnings: $30 monthly plus $1 per forwarded healthcare email.\n\n"
        
        response += "## Patient Conversation Recording (HCP-Patient Dialogue)\n"
        response += f"{WIKI_DATA['products']['hcp_pt']['description']}\n"
        response += "Earnings: Up to $1000 per month for recording patient conversations.\n\n"
        
        response += "## Surveys and Other Research Opportunities\n"
        response += f"{WIKI_DATA['products']['hcp_surveys']['description']}\n"
        response += "Earnings: $50-$200 per survey, with additional opportunities for chart reviews and interviews.\n\n"
        
        response += "Would you like to learn more about any of these specific features?"
        
        if user_id:
            CONVERSATION_CONTEXT[user_id]["follow_up_offered"] = True
            
        return response
    
    # Create or update conversation context
    if user_id:
        if user_id not in CONVERSATION_CONTEXT:
            CONVERSATION_CONTEXT[user_id] = {"last_product": None, "last_query": "", "follow_up_offered": False}
        
        context = CONVERSATION_CONTEXT[user_id]
        
        # Check if this is a follow-up to a previous question
        if context["follow_up_offered"] and user_query.lower() in ["yes", "yeah", "sure", "okay", "yep", "y"]:
            if context["last_product"]:
                # They want to know how to get started with the previously discussed product
                return format_product_info(context["last_product"], "how_to")
            else:
                # They want to know more about ZoomRx products in general
                products_overview = "Here are the main products and services ZoomRx offers:\n\n"
                for product_id, product in WIKI_DATA["products"].items():
                    products_overview += f"• {product['name']}: {product['description']}\n\n"
                return products_overview
    
    # Generate a response based on the query
    response = ""
    
    # Handle product listing queries specifically - we check this first for comparison queries
    if question_type == "products":
        # Check which product to exclude in comparison queries
        excluded_product = None
        
        if is_comparison_query:
            if "survey" in query_lower:
                excluded_product = "hcp_surveys"
                print(f"Excluding 'hcp_surveys' from response")
            elif "patient" in query_lower or "dialogue" in query_lower or "dialog" in query_lower:
                excluded_product = "hcp_pt"
                print(f"Excluding 'hcp_pt' from response")
            elif "advisory" in query_lower or "board" in query_lower:
                excluded_product = "advisory_boards" 
                print(f"Excluding 'advisory_boards' from response")
        
        if is_comparison_query and excluded_product:
            # Show all products EXCEPT the excluded one
            response = f"ZoomRx offers the following products and services in addition to {WIKI_DATA['products'][excluded_product]['name']}:\n\n"
            
            # Count how many products we're including
            included_products = [p for p in WIKI_DATA["products"] if p != excluded_product]
            print(f"Including {len(included_products)} products: {included_products}")
            
            for product_id, product in WIKI_DATA["products"].items():
                if product_id != excluded_product:
                    response += f"## {product['name']}\n"
                    response += f"{product['description']}\n\n"
                    response += "Key benefits:\n"
                    for benefit in product['benefits'][:3]:  # Show top 3 benefits
                        response += f"• {benefit}\n"
                    response += f"\nTypical earnings: {product['earnings'].split('.')[0]}.\n\n"
        else:
            # Standard product listing - show all products
            response = "ZoomRx offers the following products and services for healthcare professionals:\n\n"
            
            for product_id, product in WIKI_DATA["products"].items():
                response += f"## {product['name']}\n"
                response += f"{product['description']}\n\n"
                response += "Key benefits:\n"
                for benefit in product['benefits'][:3]:  # Show top 3 benefits
                    response += f"• {benefit}\n"
                response += f"\nTypical earnings: {product['earnings'].split('.')[0]}.\n\n"
        
        response += "Would you like to learn more about any specific product?"
        
        if user_id:
            CONVERSATION_CONTEXT[user_id]["last_product"] = None
            CONVERSATION_CONTEXT[user_id]["follow_up_offered"] = True
    
    # If specific products are mentioned (and we're not doing a comparison), focus on those
    elif mentioned_products:
        product_id = mentioned_products[0]  # Take the first mentioned product for now
        
        if user_id:
            CONVERSATION_CONTEXT[user_id]["last_product"] = product_id
        
        # If this is a feature-based query about a specific product, make it clear which product we're talking about
        if question_type == "features":
            return format_product_info(product_id, None)  # Use full response for feature queries
        else:  
            response = format_product_info(product_id, question_type)
        
        # Add a follow-up offer if appropriate
        if question_type in ["what_is", "benefits", "features"]:
            if user_id:
                CONVERSATION_CONTEXT[user_id]["follow_up_offered"] = True
            response += f"\n\n{generate_follow_up(product_id, question_type)}"
    
    # Handle general questions about ZoomRx
    elif "zoomrx" in query_lower or question_type == "general":
        response = get_general_info(question_type)
        
        # Add a follow-up for general queries
        if user_id:
            CONVERSATION_CONTEXT[user_id]["last_product"] = None
            CONVERSATION_CONTEXT[user_id]["follow_up_offered"] = True
        response += f"\n\n{generate_follow_up(None, question_type)}"
    
    # For more specific questions without a clear product
    else:
        # Try to match with a section of general info
        if question_type == "earnings":
            response = get_general_info("earnings")
            if user_id:
                CONVERSATION_CONTEXT[user_id]["follow_up_offered"] = True
        elif question_type == "eligibility":
            response = get_general_info("eligibility")
        elif question_type == "time_commitment":
            response = get_general_info("time_commitment")
        elif question_type == "how_to":
            response = "To get started with ZoomRx, you'll need to:\n\n" + \
                      "1. Create an account at zoomrx.com\n" + \
                      "2. Complete your profile with your professional information\n" + \
                      "3. Verify your healthcare credentials\n" + \
                      "4. Browse available opportunities in your dashboard"
        else:
            # Fallback to general overview            
            response = "ZoomRx offers several ways for healthcare professionals to participate and earn:\n\n"
            for product_id, product in WIKI_DATA["products"].items():
                response += f"• {product['name']}: {product['description']}\n\n"
            response += generate_follow_up(None, question_type)
            
            if user_id:
                CONVERSATION_CONTEXT[user_id]["follow_up_offered"] = True
    
    # Update conversation context
    if user_id:
        CONVERSATION_CONTEXT[user_id]["last_query"] = user_query
    
    # Print the first 100 characters of the response for debugging
    print(f"Generated response (first 100 chars): {response[:100]}")
    
    return response

# Testing
if __name__ == "__main__":
    test_queries = [
        "What is ZoomRx?",
        "Tell me about HCP surveys",
        "How much can I earn with advisory boards?",
        "How do I get started with patient referrals?",
        "What are the benefits of using ZoomRx?",
        "Who is eligible to participate?"
    ]
    
    print("\n--- Testing ZoomRx Wiki Tool ---\n")
    
    for q in test_queries:
        print(f"User Query: {q}")
        response = handle_zoomrx_wiki_query(q, user_id="test_user")
        print(f"Response: {response}\n{'-'*80}\n") 