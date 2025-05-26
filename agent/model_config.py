"""
Model Configuration for Agastya LLM System

This module provides centralized configuration for which OpenAI models 
should be used for different tasks in the Agastya workflow.
"""
import os
from dotenv import load_dotenv

# Load environment variables that might override defaults
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

# Default model configuration
MODEL_CONFIG = {
    # Task-specific model assignments
    "intent_classification": os.getenv("INTENT_CLASSIFICATION_MODEL", "o4-mini"),
    "research_synthesis": os.getenv("RESEARCH_SYNTHESIS_MODEL", "gpt-4-turbo"),
    "panel_response": os.getenv("PANEL_RESPONSE_MODEL", "gpt-4.1"),
    "conference_info": os.getenv("CONFERENCE_INFO_MODEL", "gpt-4.1"),
    "zoomrx_wiki_response": os.getenv("ZOOMRX_WIKI_MODEL", "o4-mini"),
    "final_response_generation": os.getenv("FINAL_RESPONSE_MODEL", "gpt-4.1"),
    
    # Embedding model
    "embeddings": os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"),
    
    # Fallback model (used if a specific task model is not configured)
    "default": os.getenv("DEFAULT_MODEL", "o3"),
}

def get_model_for_task(task):
    """
    Returns the appropriate model name for a given task.
    
    Args:
        task (str): The task identifier (e.g., "intent_classification")
        
    Returns:
        str: The model name to use for the task
    """
    return MODEL_CONFIG.get(task, MODEL_CONFIG["default"]) 