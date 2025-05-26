import os
import json
import logging
import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to PYTHONPATH if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to PYTHONPATH")

# Load environment variables first
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded .env from {dotenv_path}")
else:
    logger.warning(f".env file not found at {dotenv_path}")

# Import backend functions (after loading env vars)
try:
    from tools.user_info import get_user_name
    from agent.langgraph_flow import app as agastya_llm_app
    logger.info("Successfully imported backend modules")
except Exception as e:
    logger.error(f"Failed to import backend modules: {str(e)}")
    logger.error(traceback.format_exc())
    raise ImportError(f"Critical error: Failed to import backend modules: {str(e)}")

# Initialize Flask app
app = Flask(__name__)
# Allow access from any origin for API endpoints (needed for ngrok and other tunneling services)
CORS(app, resources={r"/api/*": {"origins": "*"}})
logger.info("Flask app initialized with CORS")

# Store active sessions
active_sessions = {}

@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_info(user_id):
    """
    Get user information based on user ID
    Returns first and last name from the database
    """
    logger.info(f"Received request for user_id: {user_id}")
    try:
        first_name, last_name = get_user_name(user_id)
        logger.info(f"Retrieved user info for {user_id}: {first_name} {last_name}")
        
        return jsonify({
            "firstName": first_name,
            "lastName": last_name
        })
    except Exception as e:
        logger.error(f"Error retrieving user info for {user_id}: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "message": "Failed to retrieve user information"
        }), 500

@app.route('/api/chat/init', methods=['POST'])
def init_chat_session():
    """
    Initialize a chat session for a user
    """
    data = request.json
    user_id = data.get('userId')
    
    if not user_id:
        logger.warning("Chat session init request missing userId")
        return jsonify({"error": "User ID is required"}), 400
    
    try:
        # Store the user ID in active sessions
        session_id = f"session_{user_id}_0" # Always use a consistent session ID for the same user
        active_sessions[session_id] = {
            "userId": user_id,
            "chatHistory": [], # Empty chat history to start
            "previousIntent": None,
            "conversationTopic": None
        }
        
        logger.info(f"Initialized chat session {session_id} for user {user_id}")
        return jsonify({
            "sessionId": session_id,
            "message": "Chat session initialized successfully"
        })
    except Exception as e:
        logger.error(f"Failed to initialize chat session for user {user_id}: {e}", exc_info=True)
        return jsonify({
            "error": str(e),
            "message": "Failed to initialize chat session"
        }), 500

@app.route('/api/chat/message', methods=['POST'])
def process_message():
    """
    Process a chat message using the LangGraph flow
    """
    data = request.json
    message = data.get('message')
    user_id = data.get('userId')
    
    if not message:
        logger.warning("Chat message request missing message")
        return jsonify({"error": "Message is required"}), 400
    if not user_id:
        logger.warning("Chat message request missing userId")
        return jsonify({"error": "User ID is required"}), 400
    
    try:
        # Get the chat history from the session or create an empty one
        session_id = f"session_{user_id}_0"  # Use consistent session ID for the same user
        if session_id not in active_sessions:
            logger.info(f"Creating new session {session_id} for user {user_id}")
            active_sessions[session_id] = {
                "userId": user_id,
                "chatHistory": [],
                "previousIntent": None,
                "conversationTopic": None
            }
        
        session_data = active_sessions[session_id]
        chat_history = session_data.get("chatHistory", [])
        previous_intent = session_data.get("previousIntent")
        conversation_topic = session_data.get("conversationTopic")
        
        logger.info(f"User {user_id} session state - Previous intent: {previous_intent}, Conversation topic: {conversation_topic}")
        logger.info(f"Chat history length: {len(chat_history)}")
        
        # Add the current message to chat history
        chat_history.append(HumanMessage(content=message))
        
        # Run the LangGraph app with full state
        inputs = {
            "userInput": message,
            "userId": user_id,
            "chatHistory": chat_history,
            "previousIntent": previous_intent,
            "conversationTopic": conversation_topic
        }
        
        logger.info(f"Processing message from user {user_id}: {message[:50]}...")
        response = agastya_llm_app.invoke(inputs)
        
        # Extract required data from the response
        answer = response.get("finalAnswer", "No response generated")
        intent = response.get("classifiedIntent", "unknown")
        
        # Update the session with new state
        session_data["chatHistory"] = response.get("chatHistory", chat_history)
        session_data["previousIntent"] = intent
        session_data["conversationTopic"] = response.get("conversationTopic", conversation_topic)
        
        logger.info(f"Updated state - Intent: {intent}, History length: {len(session_data['chatHistory'])}")
        logger.info(f"Generated response with intent {intent}: {answer[:50]}...")
        
        # Add the AI response to chat history if not already done by LangGraph
        if len(session_data["chatHistory"]) == len(chat_history):
            session_data["chatHistory"].append(AIMessage(content=answer))
            logger.info("Added AI response to chat history")
        
        return jsonify({
            "answer": answer,
            "intent": intent
        })
    except Exception as e:
        logger.error(f"Failed to process message for user {user_id}: {e}", exc_info=True)
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "message": "Failed to process message"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({
        "status": "ok",
        "message": "API server is running"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting API server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True) 