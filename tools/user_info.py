import os
import logging
from db.sql_connector import execute_sql_query
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection details (from environment variables)
DB_HOST = os.getenv("PANEL_DB_HOST")
DB_USER = os.getenv("PANEL_DB_USER")
DB_PASSWORD = os.getenv("PANEL_DB_PASSWORD")
DB_NAME = os.getenv("PANEL_DB_NAME")

# Connection details dictionary
DB_CONNECTION_DETAILS = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME
}

def get_user_name(user_id: int) -> tuple:
    """
    Fetch the user's name from the database.
    
    Args:
        user_id: The unique identifier of the user
    
    Returns:
        tuple: (first_name, last_name)
    """
    if not all(DB_CONNECTION_DETAILS.values()):
        logger.error("Database configuration is incomplete. Cannot fetch user name.")
        return None, None
    
    query = """
        SELECT first_name, last_name
        FROM users
        WHERE id = %s
    """
    
    try:
        results = execute_sql_query(query, (user_id,), conn_details=DB_CONNECTION_DETAILS)
        if results and len(results) > 0:
            return results[0].get("first_name"), results[0].get("last_name")
        else:
            logger.warning(f"No user found with ID: {user_id}")
            return None, None
    except Exception as e:
        logger.error(f"Error fetching user name for ID {user_id}: {e}", exc_info=True)
        return None, None

def format_greeting(user_id: int) -> str:
    """
    Format a greeting message for the user using their name.
    
    Args:
        user_id: The unique identifier of the user
    
    Returns:
        str: Formatted greeting message
    """
    first_name, last_name = get_user_name(user_id)
    
    if first_name and last_name:
        return f"Hello Dr. {last_name}! How can I help you today?"
    else:
        return f"Hello User {user_id}! How can I help you today?"

# For testing
if __name__ == "__main__":
    test_id = 123  # Replace with a valid user ID from your database
    print(format_greeting(test_id)) 