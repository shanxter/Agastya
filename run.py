import os
import sys
import logging
from dotenv import load_dotenv

# --- Debug information (useful for troubleshooting import issues) ---
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
# --- End debug information ---

# --- .env loading (MUST be at the top before other project imports that might use env vars) ---
# This loads variables from Agastya/.env
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    # print(f"DEBUG (run.py): Loaded .env from {dotenv_path}") # Optional for debugging
else:
    # print(f"DEBUG (run.py): .env file not found at {dotenv_path}. Relying on system env vars.") # Optional
    pass
# --- END .env loading ---

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Import the LangGraph application ---
# This assumes your compiled LangGraph app is named 'app' in agent/langgraph_flow.py
try:
    from agent.langgraph_flow import app as agastya_llm_app
    logger.info("Successfully imported Agastya LLM application graph.")
except ImportError as e:
    logger.critical(f"Failed to import 'app' from 'agent.langgraph_flow'. Error: {e}")
    logger.critical("Ensure 'agent/langgraph_flow.py' exists and is correctly structured.")
    exit()
except Exception as e:
    logger.critical(f"An unexpected error occurred during import of 'agent.langgraph_flow': {e}", exc_info=True)
    exit()

# Import user info tools
try:
    from tools.user_info import format_greeting
    logger.info("Successfully imported user info tools.")
except ImportError as e:
    logger.warning(f"Failed to import from 'tools.user_info'. Will use default greeting. Error: {e}")


# --- Main Interaction Loop ---
def main_interaction_loop():
    logger.info("Initializing Agastya HCP LLM Assistant...")
    print("\nWelcome to the Agastya HCP LLM Assistant!")
    print("I can help with medical research, panel data queries, and conference information.")
    print("Type 'exit' or 'quit' to end the session.\n")

    # --- User Identification (Placeholder) ---
    # In a real application, user_id would come from a login/authentication system.
    # For now, we'll use a hardcoded example or prompt for it.
    try:
        user_id_input = input("Enter your User ID (e.g., 78060, or press Enter for default 123): ")
        current_user_id = int(user_id_input) if user_id_input.strip() else 123 # Default if empty
        logger.info(f"Using User ID: {current_user_id} for this session.")
    except ValueError:
        current_user_id = 123 # Default if input is not a valid integer
        logger.warning(f"Invalid User ID input. Defaulting to User ID: {current_user_id}.")
    
    # Use the new greeting function
    try:
        greeting = format_greeting(current_user_id)
        print(greeting)
    except Exception as e:
        logger.error(f"Error formatting greeting: {e}. Using default greeting.")
        print(f"Hello User {current_user_id}!")
    # --- End User Identification ---

    chat_history = [] # Stores LangChain BaseMessage objects (HumanMessage, AIMessage)

    while True:
        try:
            user_query = input("\nYou: ")
            if user_query.lower() in ['exit', 'quit']:
                print("Agastya: Goodbye!")
                break
            if not user_query.strip():
                continue

            # Prepare input for the LangGraph app
            # The AgentState in langgraph_flow.py expects 'userInput', 'userId', and 'chatHistory'
            graph_input = {
                "userInput": user_query,
                "userId": current_user_id,
                "chatHistory": chat_history 
            }

            print("Agastya is thinking...")
            
            final_graph_state = None
            full_response_chunks = [] # To accumulate streamed response if using streaming LLM

            # Invoke the LangGraph app.
            # Using app.stream() allows observing intermediate states or streaming tokens.
            # For a simpler final response, you could use: final_graph_state = agastya_llm_app.invoke(graph_input)
            
            for event_chunk in agastya_llm_app.stream(graph_input, stream_mode="values"):
                # `event_chunk` will be the full AgentState at each step.
                # You can inspect intermediate states here for debugging if needed:
                # logger.debug(f"--- Graph State Update --- \n{event_chunk}")
                final_graph_state = event_chunk # The last event contains the final state

                # If your final LLM node supports streaming tokens for 'finalAnswer':
                # if "finalAnswer" in event_chunk and event_chunk["finalAnswer"] not in full_response_chunks:
                #     new_token = event_chunk["finalAnswer"].replace("".join(full_response_chunks), "")
                #     print(new_token, end="", flush=True) # Print token immediately
                #     full_response_chunks.append(new_token)
            
            # print() # Newline after streaming potentially

            if final_graph_state and final_graph_state.get("finalAnswer"):
                ai_response = final_graph_state["finalAnswer"]
                print(f"Agastya: {ai_response}")
                
                # Update chat history from the final state for the next turn
                chat_history = final_graph_state.get("chatHistory", [])
                
                # Optional: Limit chat history size to prevent overly long contexts
                if len(chat_history) > 10: # Keep last 5 conversation pairs (10 messages)
                    logger.debug(f"Truncating chat history from {len(chat_history)} messages.")
                    chat_history = chat_history[-10:]
            else:
                error_message = "Agastya: I encountered an issue and couldn't formulate a response. Please try again."
                print(error_message)
                # Add a generic error to history to maintain conversation flow
                from langchain_core.messages import HumanMessage, AIMessage # Ensure imports if not global
                chat_history.append(HumanMessage(content=user_query))
                chat_history.append(AIMessage(content="I had an issue processing that request."))


        except KeyboardInterrupt:
            print("\nAgastya: Session interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the interaction loop: {e}", exc_info=True)
            print("Agastya: A system error occurred. Please try again later.")
            # Optionally, break or attempt to recover

if __name__ == "__main__":
    # Critical check for OpenAI API Key before starting
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("CRITICAL: OPENAI_API_KEY environment variable not found.")
        logger.critical("Please ensure it is set in your .env file (in the Agastya/ root) or system environment.")
        print("Application cannot start without OPENAI_API_KEY. Exiting.")
    else:
        logger.info("OpenAI API Key found. Starting application.")
        main_interaction_loop()