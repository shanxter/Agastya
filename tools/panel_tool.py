import os
import logging
from datetime import datetime, timedelta
import re # For more advanced parsing if needed, not heavily used in this version
from typing import Tuple, Dict, Any, Optional, List, Union

# Import the actual SQL execution function from your db module
from db.sql_connector import execute_sql_query

# --- .env loading for DB credentials ---
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
# --- END .env loading ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database Connection Details (from Environment Variables) ---
DB_HOST = os.getenv("PANEL_DB_HOST")
DB_USER = os.getenv("PANEL_DB_USER")
DB_PASSWORD = os.getenv("PANEL_DB_PASSWORD")
DB_NAME = os.getenv("PANEL_DB_NAME")

# This dictionary will be passed to the actual execute_sql_query function
DB_CONNECTION_DETAILS = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME
}

# Load database schema context for LLM-generated queries
SCHEMA_CONTEXT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db_schema_context.txt')
DB_SCHEMA_CONTEXT = ""
if os.path.exists(SCHEMA_CONTEXT_PATH):
    try:
        with open(SCHEMA_CONTEXT_PATH, 'r') as f:
            DB_SCHEMA_CONTEXT = f.read()
        logger.info(f"Loaded database schema context from {SCHEMA_CONTEXT_PATH}")
    except Exception as e:
        logger.error(f"Error loading schema context: {e}")
else:
    logger.warning(f"Schema context file not found at {SCHEMA_CONTEXT_PATH}")

# --- Static FAQ Answers ---
STATIC_ANSWERS = {
    "change email id": "To change your email ID, please go to your Profile section on our website and look for the 'Edit Profile' or 'Account Settings' option. You should be able to update your email address there. If you face any issues, please contact support.",
    "update profile": "You can update your profile information, including personal details and payment preferences, by navigating to the 'Profile' or 'My Account' section after logging into our platform.",
    "payment methods": "We offer several payment methods, typically including PayPal and direct bank transfers, depending on your region. Please check the 'Payments' or 'Rewards' section for details specific to your account.",
    "forgot password": "If you've forgotten your password, please use the 'Forgot Password?' link on the login page. You'll receive an email with instructions to reset it."
    # Add more static Q&A pairs here, using lowercase triggers
}

def _calculate_date_range(time_period_keyword: str) -> tuple[str | None, str | None]:
    """
    Calculates start and end dates based on keywords like "last month", "last week", "last year".
    Also handles specific months (e.g., "April 2025") and years (e.g., "2025").
    Returns (start_date_str, end_date_str) in 'YYYY-MM-DD' format, or (None, None).
    """
    today = datetime.now()
    start_date, end_date = None, None
    time_period_lower = time_period_keyword.lower()

    # Check for "all time" queries
    if "all time" in time_period_lower or "all-time" in time_period_lower or "lifetime" in time_period_lower:
        # Use a very early date as start_date and today as end_date
        start_date = datetime(2000, 1, 1)  # Using year 2000 as a reasonable "beginning of time" for the panel
        end_date = today
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    # Standard relative time periods
    if "last month" in time_period_lower:
        first_day_current_month = today.replace(day=1)
        last_day_last_month = first_day_current_month - timedelta(days=1)
        start_date = last_day_last_month.replace(day=1)
        end_date = last_day_last_month
    elif "last year" in time_period_lower:
        start_date = today.replace(year=today.year - 1, month=1, day=1)
        end_date = today.replace(year=today.year - 1, month=12, day=31)
    elif "last week" in time_period_lower:
        start_date = today - timedelta(days=today.weekday() + 7) # Start of last week (Monday)
        end_date = start_date + timedelta(days=6) # End of last week (Sunday)
    elif "this month" in time_period_lower: # From start of this month until today
        start_date = today.replace(day=1)
        end_date = today
    elif "this year" in time_period_lower: # From start of this year until today
        start_date = today.replace(month=1, day=1)
        end_date = today
    elif "this week" in time_period_lower: # From start of this week (Monday) until today
        start_date = today - timedelta(days=today.weekday())
        end_date = today
    else:
        # Check for specific year patterns like "2025" or "in 2025"
        year_pattern = r'(?:^|\s)(\d{4})(?:\s|$)' 
        year_match = re.search(year_pattern, time_period_lower)
        
        # Check for specific month and year patterns like "April 2025" or "Apr 2025"
        month_year_pattern = r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})'
        month_year_match = re.search(month_year_pattern, time_period_lower)
        
        if month_year_match:
            # Handle "Month Year" format
            month_name = month_year_match.group(1).lower()
            year = int(month_year_match.group(2))
            
            # Convert month name to number
            month_dict = {
                'january': 1, 'jan': 1,
                'february': 2, 'feb': 2,
                'march': 3, 'mar': 3,
                'april': 4, 'apr': 4,
                'may': 5,
                'june': 6, 'jun': 6,
                'july': 7, 'jul': 7,
                'august': 8, 'aug': 8,
                'september': 9, 'sep': 9,
                'october': 10, 'oct': 10,
                'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            month_num = month_dict.get(month_name, 1)
            
            # Create date range for that specific month
            start_date = datetime(year=year, month=month_num, day=1)
            # Get last day of month
            if month_num == 12:
                end_date = datetime(year=year+1, month=1, day=1) - timedelta(days=1)
            else:
                end_date = datetime(year=year, month=month_num+1, day=1) - timedelta(days=1)
        elif year_match:
            # Handle specific year like "2025"
            year = int(year_match.group(1))
            start_date = datetime(year=year, month=1, day=1)
            end_date = datetime(year=year, month=12, day=31)
    
    # Handle "last X days/weeks/months" with regex
    last_x_pattern = r'last\s+(\d+)\s+(day|days|week|weeks|month|months)'
    last_x_match = re.search(last_x_pattern, time_period_lower)
    if last_x_match:
        number = int(last_x_match.group(1))
        unit = last_x_match.group(2)
        end_date = today
        
        if unit in ['day', 'days']:
            start_date = today - timedelta(days=number)
        elif unit in ['week', 'weeks']:
            start_date = today - timedelta(weeks=number)
        elif unit in ['month', 'months']:
            # Calculate months by manipulating year and month
            month = today.month - (number % 12)
            year_diff = number // 12
            if month <= 0:
                month += 12
                year_diff += 1
            start_date = today.replace(year=today.year - year_diff, month=month, day=1)

    return start_date.strftime("%Y-%m-%d") if start_date else None, \
           end_date.strftime("%Y-%m-%d") if end_date else None

def get_user_earnings(user_id: int, start_date: str, end_date: str) -> str:
    """Fetches total earnings for a user over a specified period."""
    if not all(DB_CONNECTION_DETAILS.values()):
        return "Database configuration is incomplete. Cannot fetch earnings."

    # Check if the requested date range is in the future
    today = datetime.now()
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Special handling for 2025: Allow data from Jan 1, 2025 to current date
    if start_date_dt.year == 2025:
        # For 2025, use Jan 1, 2025 to today
        start_date = f"2025-01-01"
        end_date = today.strftime("%Y-%m-%d")
    # For years beyond 2025, still show the future year message
    elif start_date_dt > today:
        year = start_date_dt.year
        return f"I don't have earnings data for {year} yet as it's in the future. I can only provide historical earnings data. Would you like to see your earnings for the current year instead?"

    query = """
        SELECT SUM(t.amount) AS total_earnings
        FROM users_waves uw
        LEFT JOIN earnings e ON uw.id = e.users_wave_id
        JOIN transactions t ON t.id = e.transaction_id
        WHERE uw.user_id = %s
          AND uw.completed_date >= %s 
          AND uw.completed_date <= %s; 
    """
    params = (user_id, start_date, end_date)
    try:
        results = execute_sql_query(query, params, conn_details=DB_CONNECTION_DETAILS)
        if results and results[0].get("total_earnings") is not None:
            total_earnings = results[0]["total_earnings"]
            return f"Your total earnings from {start_date} to {end_date} were: ${total_earnings:.2f}."
        else:
            return f"No earnings found for you from {start_date} to {end_date}."
    except Exception as e:
        logger.error(f"SQL Error in get_user_earnings for user {user_id}: {e}", exc_info=True)
        return "An error occurred while fetching your earnings."

def get_completed_surveys(user_id: int, start_date: str, end_date: str) -> str:
    """Fetches distinct completed survey titles for a user over a period."""
    if not all(DB_CONNECTION_DETAILS.values()):
        return "Database configuration is incomplete. Cannot fetch completed surveys."
    
    # Check if the requested date range is in the future
    today = datetime.now()
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Special handling for 2025: Allow data from Jan 1, 2025 to current date
    if start_date_dt.year == 2025:
        # For 2025, use Jan 1, 2025 to today
        start_date = f"2025-01-01"
        end_date = today.strftime("%Y-%m-%d")
    # For years beyond 2025, still show the future year message
    elif start_date_dt > today:
        year = start_date_dt.year
        return f"I don't have survey data for {year} yet as it's in the future. I can only provide historical survey data. Would you like to see your completed surveys for the current year instead?"
        
    query = """
        SELECT DISTINCT(l.surveyls_display_title)
        FROM surveys_users su
        JOIN waves w ON w.survey_id = su.survey_id
        JOIN surveys s ON su.survey_id = s.id
        JOIN lime_surveys_languagesettings l ON l.surveyls_survey_id = su.survey_id
        JOIN users_waves uw ON uw.wave_id = w.id
        WHERE uw.user_id = %s 
          AND uw.status = 1 
          AND uw.completed_date >= %s
          AND uw.completed_date <= %s;
    """
    params = (user_id, start_date, end_date)
    try:
        results = execute_sql_query(query, params, conn_details=DB_CONNECTION_DETAILS)
        if results:
            survey_titles = [row["surveyls_display_title"] for row in results if "surveyls_display_title" in row]
            if survey_titles:
                return f"Surveys you completed from {start_date} to {end_date}:\n- " + "\n- ".join(survey_titles)
            else:
                 return f"No surveys found as completed by you from {start_date} to {end_date}."
        else:
            return f"No surveys found as completed by you from {start_date} to {end_date}."
    except Exception as e:
        logger.error(f"SQL Error in get_completed_surveys for user {user_id}: {e}", exc_info=True)
        return "An error occurred while fetching your completed surveys."

def get_last_participation_date(user_id: int) -> str:
    """Fetches the last participation (completed) date for a user."""
    if not all(DB_CONNECTION_DETAILS.values()):
        return "Database configuration is incomplete. Cannot fetch last participation date."

    query = """
        SELECT MAX(uw.completed_date) AS last_participation_date
        FROM users_waves uw
        WHERE uw.user_id = %s AND uw.status = 1; 
    """
    params = (user_id,)
    try:
        results = execute_sql_query(query, params, conn_details=DB_CONNECTION_DETAILS)
        if results and results[0].get("last_participation_date") is not None:
            last_date = results[0]["last_participation_date"]
            # Assuming the date comes back as a datetime object or a string that can be parsed
            if isinstance(last_date, datetime):
                last_date_str = last_date.strftime('%Y-%m-%d')
            else:
                last_date_str = str(last_date).split(" ")[0] # Get YYYY-MM-DD part if it's a string with time
            return f"Your last participation (completed survey) date was: {last_date_str}."
        else:
            return "No participation records found to determine your last participation date."
    except Exception as e:
        logger.error(f"SQL Error in get_last_participation_date for user {user_id}: {e}", exc_info=True)
        return "An error occurred while fetching your last participation date."

def get_time_earned_stats(user_id: int, start_date: str, end_date: str) -> str:
    """Fetches total time spent on surveys for a user over a period."""
    if not all(DB_CONNECTION_DETAILS.values()):
        return "Database configuration is incomplete. Cannot fetch time earned statistics."

    # Check if this is an "all time" query
    is_all_time = False
    if start_date == "2000-01-01":  # Our convention for "all time"
        is_all_time = True
    
    # Check if the requested date range is in the future (skip for "all time" queries)
    if not is_all_time:
        today = datetime.now()
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Special handling for 2025: Allow data from Jan 1, 2025 to current date
        if start_date_dt.year == 2025:
            # For 2025, use Jan 1, 2025 to today
            start_date = f"2025-01-01"
            end_date = today.strftime("%Y-%m-%d")
        # For years beyond 2025, still show the future year message
        elif start_date_dt > today:
            year = start_date_dt.year
            return f"I don't have time spent data for {year} yet as it's in the future. I can only provide historical data. Would you like to see your time statistics for the current year instead?"

    # Base query with required joins and filters
    query = """
        SELECT 
            SUM(uwd.time_taken) / 60.0 AS total_time_minutes,
            AVG(uwd.time_taken) / 60.0 AS avg_time_minutes,
            COUNT(DISTINCT uw.id) As total_surveys_completed
        FROM users_waves uw 
        JOIN waves w ON w.id = uw.wave_id
        JOIN users_wave_details uwd ON uwd.id = uw.id 
        JOIN lime_surveys_languagesettings lsl ON lsl.surveyls_survey_id = w.survey_id
        WHERE uw.user_id = %s
          AND uw.status = 1 
          AND w.survey_id NOT IN (SELECT sa.survey_id FROM survey_attributes sa WHERE sa.attribute = 'paired_common_survey_id')
    """
    
    # Add date constraints only if not an "all time" query
    if not is_all_time:
        query += """
          AND uw.completed_date >= %s 
          AND uw.completed_date <= %s
        """
        params = (user_id, start_date, end_date)
    else:
        params = (user_id,)
    
    try:
        results = execute_sql_query(query, params, conn_details=DB_CONNECTION_DETAILS)
        if results and results[0].get("total_time_minutes") is not None:
            stats = results[0]
            total_time = stats.get("total_time_minutes", 0)
            avg_time = stats.get("avg_time_minutes", 0)
            count_surveys = stats.get("total_surveys_completed", 0)
            
            time_period_text = "all time" if is_all_time else f"the period {start_date} to {end_date}"
            
            return (f"For {time_period_text}, you:\n"
                    f"- Spent a total of {total_time:.2f} minutes on surveys.\n"
                    f"- Averaged {avg_time:.2f} minutes per survey.\n"
                    f"- Completed {count_surveys} surveys (excluding paired common surveys).")
        else:
            time_period_text = "all time" if is_all_time else f"from {start_date} to {end_date}"
            return f"No time records found for you for {time_period_text} (excluding paired common surveys)."
    except Exception as e:
        logger.error(f"SQL Error in get_time_earned_stats for user {user_id}: {e}", exc_info=True)
        return "An error occurred while fetching your time earned statistics."

def generate_and_execute_query(user_query: str, user_id: int, time_period: str = None, 
                               start_date: str = None, end_date: str = None) -> str:
    """
    Uses the database schema context to generate and execute an appropriate SQL query
    based on the user's natural language query.
    
    This is intended for complex or specific queries that aren't handled by the standard functions.
    """
    from openai import OpenAI
    client = OpenAI()
    
    if not DB_SCHEMA_CONTEXT:
        return "Unable to generate SQL query due to missing database schema context."
    
    # Construct a prompt for the LLM with the necessary context
    time_context = ""
    is_all_time_query = time_period and "all time" in time_period.lower()
    
    if start_date and end_date and not is_all_time_query:
        time_context = f"The query is for the time period from {start_date} to {end_date}."
    elif is_all_time_query:
        time_context = (
            "This is an 'all time' query, so DO NOT include date range filters in the WHERE clause. "
            "However, KEEP ALL OTHER IMPORTANT FILTERS intact, especially exclusions like: "
            "AND w.survey_id NOT IN (SELECT sa.survey_id FROM survey_attributes sa WHERE sa.attribute = 'paired_common_survey_id')"
        )
    elif time_period:
        time_context = f"The query is about time period: '{time_period}'."
        
    system_prompt = f"""You are a SQL query generator for a panel management system. 
Generate a SQL query to answer the user's question based on the database schema provided.
Only return the SQL query, no explanations or markdown.
Do not wrap the query in code blocks or backticks.
The user ID is {user_id} and should be used in the WHERE clause where appropriate.
{time_context}

IMPORTANT: When joining users_wave_details to users_waves, use "JOIN users_wave_details uwd ON uwd.id = uw.id" NOT "uwd.user_wave_id = uw.id".

The query must be safe and only perform read operations (SELECT).
Do not use any DDL or DML operations (CREATE, INSERT, UPDATE, DELETE, DROP, etc).
Always use parameterized queries with %s placeholders for values that will be injected.

Here is the database schema information:
{DB_SCHEMA_CONTEXT}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Or your preferred model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a SQL query to answer: {user_query}"}
            ],
            max_tokens=500,
            temperature=0
        )
        
        sql_query = response.choices[0].message.content.strip()
        
        # Strip any markdown code block formatting that might be in the response
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Fix common error: replace user_wave_id with users_wave_id
        sql_query = sql_query.replace("user_wave_id", "users_wave_id")
        
        # Add basic safety checks
        sql_lower = sql_query.lower()
        if any(keyword in sql_lower for keyword in ["insert", "update", "delete", "drop", "alter", "truncate", "create"]):
            logger.error(f"Generated query contains prohibited operations: {sql_query}")
            return "Error: Generated query contains prohibited operations."
            
        logger.info(f"Generated SQL query: {sql_query}")
        
        # Determine the appropriate parameters based on query
        params = []
        if "%s" in sql_query:
            params = [user_id]  # Always add user_id as first parameter
            
            # Add date parameters if they exist and are in the query
            if sql_lower.count("%s") > 1 and start_date and end_date and not is_all_time_query:
                params.extend([start_date, end_date])
                
        # Execute the query
        results = execute_sql_query(sql_query, tuple(params), conn_details=DB_CONNECTION_DETAILS)
        
        # Format the results nicely - this is a simple formatter
        if not results:
            return "No results found for your query."
        
        # For single row queries with one main result
        if len(results) == 1 and len(results[0]) == 1:
            key = list(results[0].keys())[0]
            value = results[0][key]
            return f"{key.replace('_', ' ').title()}: {clean_value(value)}"
        
        # For more complex results, create a formatted report
        response = "Here's what I found:\n\n"
        
        # If it's a simple list of results (multiple rows, 1-2 columns)
        if all(len(row) <= 2 for row in results):
            for row in results:
                if len(row) == 1:
                    key = list(row.keys())[0]
                    response += f"- {clean_value(row[key])}\n"
                else:
                    # If there are two columns, assume it's a name-value pair
                    keys = list(row.keys())
                    response += f"- {clean_value(row[keys[0]])}: {clean_value(row[keys[1]])}\n"
            return response
            
        # For tabular data (multiple columns)
        else:
            # Show the first 10 rows to avoid overwhelming responses
            for i, row in enumerate(results[:10]):
                response += f"Record {i+1}:\n"
                for key, value in row.items():
                    # Clean the value - convert to string and remove HTML tags
                    if value is not None:
                        value_str = clean_value(value)
                    else:
                        value_str = "None"
                    response += f"  - {key.replace('_', ' ').title()}: {value_str}\n"
                response += "\n"
                
            if len(results) > 10:
                response += f"... and {len(results) - 10} more records."
                
            return response
            
    except Exception as e:
        logger.error(f"Error generating or executing query: {e}", exc_info=True)
        return f"An error occurred while processing your request: {str(e)}"

def handle_panel_query(user_query: str, user_id: int) -> str:
    """
    Main function for the panel tool.
    Parses the user_query to determine which SQL function to call or if it's a static FAQ.
    Extracts time period if relevant.
    `user_id` must be provided.
    """
    query_lower = user_query.lower()
    logger.info(f"Panel Tool: Handling query '{user_query}' for user_id {user_id}")

    if not user_id:
        return "Error: User ID not provided. Cannot process panel query."

    # 1. Check for Static FAQs first
    for faq_trigger, answer in STATIC_ANSWERS.items():
        if faq_trigger in query_lower:
            logger.info(f"Panel tool: Matched static FAQ for '{faq_trigger}'")
            return answer

    # 2. Determine time period from query for dynamic queries
    # If no specific time period is mentioned, default to "all time" instead of "last month"
    time_period_keyword = "all time" # Changed default to "all time"
    
    # Check for "all time" queries
    if "all time" in query_lower or "all-time" in query_lower or "lifetime" in query_lower:
        time_period_keyword = "all time"
        logger.info(f"Panel tool: Detected 'all time' query")
    else:
        # Check for specific date patterns first
        # Month-Year pattern (e.g., "April 2025" or "Apr 2025")
        month_year_pattern = r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})'
        month_year_match = re.search(month_year_pattern, query_lower)
        
        # Year patterns with multiple variations (e.g., "2025", "in 2025", "for 2025", etc.)
        year_patterns = [
            r'in\s+(\d{4})',  # "in 2025"
            r'for\s+(\d{4})',  # "for 2025"
            r'during\s+(\d{4})',  # "during 2025"
            r'about\s+.*?(\d{4})',  # "about my earnings in 2025"
            r'(?:earn|earned|earnings|income).*?(\d{4})',  # "earnings in 2025" or similar patterns
            r'(?:^|\s)(\d{4})(?:\s|$)'  # Basic 4-digit year with word boundaries (try last)
        ]
        
        # Try all patterns to find year mentions
        year_match = None
        matched_year = None
        
        for pattern in year_patterns:
            current_match = re.search(pattern, query_lower)
            if current_match:
                year_match = True
                matched_year = current_match.group(1)
                logger.info(f"Panel tool: Detected year pattern with '{pattern}': {matched_year}")
                break
        
        # "Last X days/weeks/months" pattern
        last_x_pattern = r'last\s+(\d+)\s+(day|days|week|weeks|month|months)'
        last_x_match = re.search(last_x_pattern, query_lower)
        
        # Set the time period based on detected patterns
        if month_year_match:
            time_period_keyword = f"{month_year_match.group(0)}"
            logger.info(f"Panel tool: Detected month-year pattern: {time_period_keyword}")
        elif year_match:
            time_period_keyword = matched_year
            logger.info(f"Panel tool: Detected year pattern: {time_period_keyword}")
        elif last_x_match:
            time_period_keyword = f"{last_x_match.group(0)}"
            logger.info(f"Panel tool: Detected 'last X' pattern: {time_period_keyword}")
        # Standard time periods
        elif "last month" in query_lower: time_period_keyword = "last month"
        elif "last year" in query_lower: time_period_keyword = "last year"
        elif "last week" in query_lower: time_period_keyword = "last week"
        elif "this month" in query_lower: time_period_keyword = "this month"
        elif "this year" in query_lower: time_period_keyword = "this year"
        elif "this week" in query_lower: time_period_keyword = "this week"
        else:
            logger.info(f"Panel tool: No time period specified, using default 'all time'")
            # Keep the default "all time" that's already set

    start_date, end_date = _calculate_date_range(time_period_keyword)

    # 3. Check for standard query patterns with predefined functions
    is_earnings_query = (("how much" in query_lower and "earn" in query_lower) or 
        "earnings" in query_lower or 
        "check earn" in query_lower or
        "show earn" in query_lower or
        "view earn" in query_lower or
        "tell me about earn" in query_lower or
        "tell me my earn" in query_lower or
        "know my earn" in query_lower or
        "get my earn" in query_lower)
        
    is_survey_list_query = (("surveys" in query_lower and "completed" in query_lower) or 
         ("what surveys" in query_lower and "did i complete" in query_lower) or 
         ("which surveys" in query_lower))
         
    is_last_participation_query = "last participation date" in query_lower or "when did i last participate" in query_lower
    
    is_time_stats_query = "time" in query_lower and "earn" in query_lower and not ("all time" in query_lower)
    
    # Story-related queries
    is_story_query = "story" in query_lower or "relationship with zoomrx" in query_lower or "journey" in query_lower
    
    # Handle known query patterns with predefined functions
    if is_earnings_query and not is_time_stats_query:
        if start_date and end_date:
            return get_user_earnings(user_id, start_date, end_date)
        else:
            # If no date range is specified, default to "all time" with a very early start date
            return get_user_earnings(user_id, 
                                   "2000-01-01",  # Early start date for "all time" 
                                   datetime.now().strftime("%Y-%m-%d"))
    
    elif is_time_stats_query:
        if start_date and end_date:
            return get_time_earned_stats(user_id, start_date, end_date)
        else:
            # If no date range is specified, default to "all time" with a very early start date
            return get_time_earned_stats(user_id, 
                                       "2000-01-01",  # Early start date for "all time"
                                       datetime.now().strftime("%Y-%m-%d"))
    
    elif is_survey_list_query:
        if start_date and end_date:
            return get_completed_surveys(user_id, start_date, end_date)
        else:
            # If no date range is specified, default to "all time" with a very early start date
            return get_completed_surveys(user_id, 
                                       "2000-01-01",  # Early start date for "all time"
                                       datetime.now().strftime("%Y-%m-%d"))

    elif is_last_participation_query:
        return get_last_participation_date(user_id)
        
    # 4. For any other queries, use the LLM-driven approach
    # If we can't match any standard pattern, try to generate a query
    elif DB_SCHEMA_CONTEXT:
        logger.info(f"Using LLM-driven query generation for: '{user_query}'")
        return generate_and_execute_query(user_query, user_id, time_period_keyword, start_date, end_date)

    # 5. Fallback if nothing worked
    logger.warning(f"Panel tool: No specific dynamic query matched for: '{user_query}' for user {user_id}")
    return ("I can help with questions about your panel data:\n"
            "- Earnings in a time period (e.g., 'How much did I earn in April 2025?')\n"
            "- Completed surveys (e.g., 'What surveys did I complete last month?')\n"
            "- Time spent (e.g., 'How much time did I spend on surveys in 2025?')\n"
            "- Last participation (e.g., 'When did I last participate?')\n"
            "- Your ZoomRx story (e.g., 'Tell me about my relationship with ZoomRx')\n"
            "- Or ask me any other question about your panel data\n\n"
            "Please specify a time period like 'last month', 'this year', 'April 2025', '2025', or 'all time' for time-based queries.")

def clean_value(value) -> str:
    """Clean a value by converting to string and removing HTML tags."""
    if value is None:
        return "None"
    value_str = str(value)
    # Simple HTML tag removal
    return re.sub(r'<[^>]+>', '', value_str)

# --- Example Test (for direct testing of this file) ---
if __name__ == "__main__":
    if not all(DB_CONNECTION_DETAILS.values()):
        print("WARNING: Panel database credentials (PANEL_DB_HOST, _USER, _PASSWORD, _NAME) "
              "not fully set in .env. SQL queries will use placeholders or may fail if "
              "the placeholder execute_sql_query is replaced with a real one.")
    else:
        print("Panel database credentials seem to be set. Real SQL queries will be attempted if placeholder is replaced.")

    test_user_id = 78060 # Example User ID

    print("\n--- Testing Static FAQs ---")
    print(f"Q: How do I change my email ID?\nA: {handle_panel_query('How do I change my email ID?', test_user_id)}")
    
    print("\n--- Testing Date Range Extraction ---")
    test_date_ranges = [
        "last month",
        "last year",
        "last week",
        "this month",
        "this year",
        "this week",
        "last 3 months",
        "last 30 days",
        "last 2 weeks",
        "April 2025",
        "Apr 2025",
        "January 2024",
        "Jan 2024",
        "2025",
        "in 2023",
        "all time",
        "lifetime"
    ]
    
    for date_range in test_date_ranges:
        start, end = _calculate_date_range(date_range)
        print(f"Date Range: '{date_range}'")
        print(f"  Start Date: {start}")
        print(f"  End Date: {end}")
    
    print("\n--- Testing Dynamic Queries (using placeholder SQL execution) ---")
    queries_to_test = [
        "How much did I earn last month?",
        "What are the surveys I have completed over the last year?",
        "What was my last participation date?",
        "How much time I have earned over the last week?",
        "Tell me about my earnings this month.",
        "How much did I earn in April 2025?",
        "How much did I earn in Apr 2025?",
        "What surveys did I complete in 2023?",
        "How much time did I spend on surveys in the last 3 months?",
        "Which surveys did I complete?",  # Should use default 30-day range
        "How much did I earn all time?",
        "Can you get me my earnings all time?",
        "gibberish query panel info" # Test fallback
    ]

    for q in queries_to_test:
        print(f"\nUser Query: {q}")
        response = handle_panel_query(q, test_user_id)
        print(f"Tool Response: {response}")