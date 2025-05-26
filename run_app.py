#!/usr/bin/env python
import os
import subprocess
import sys
import time
import threading
import signal
import webbrowser
import platform
from dotenv import load_dotenv

# Set up better logging
print("="*80)
print("Agastya Startup Diagnostics")
print("="*80)
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Current directory: {os.getcwd()}")
print("-"*80)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}")

# Verify Python environment
try:
    import langchain_openai
    import langchain_core
    import langgraph
    import flask
    import flask_cors
    print("Required Python packages verified: langchain, langgraph, flask")
except ImportError as e:
    print(f"Critical error: Required package missing: {e}")
    print("Please ensure you've activated the virtual environment and installed all requirements.")
    sys.exit(1)

# Define process objects for cleanup
frontend_process = None
backend_process = None

def get_venv_python():
    """Get the path to the Python executable in the virtual environment"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'win32':
        venv_python = os.path.join(base_dir, '.venv-py312', 'Scripts', 'python.exe')
    else:
        venv_python = os.path.join(base_dir, '.venv-py312', 'bin', 'python')
    
    if os.path.exists(venv_python):
        print(f"Found virtual environment Python at: {venv_python}")
        return venv_python
    else:
        print(f"Warning: Virtual environment Python not found at {venv_python}")
        print(f"Using system Python instead: {sys.executable}")
        return sys.executable

def run_backend():
    """Start the Flask API server using the virtual environment Python"""
    global backend_process
    print("Starting backend API server...")
    
    # Use the Python from the virtual environment
    venv_python = get_venv_python()
    
    # Set additional environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # Start the backend process
    backend_process = subprocess.Popen(
        [venv_python, "api_server.py"], 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    # Check if process started successfully
    if backend_process.poll() is not None:
        print(f"Error: Backend process exited immediately with code {backend_process.returncode}")
        return
    
    # Print backend output
    backend_started = False
    for line in backend_process.stdout:
        print(f"[Backend] {line.strip()}")
        
        # Check for successful startup message
        if "Starting API server" in line or "Running on" in line:
            backend_started = True

def run_frontend():
    """Start the frontend development server"""
    global frontend_process
    print("Starting frontend development server...")
    os.chdir("frontend")
    
    # On Windows, use cmd to run npm commands - this ensures we use the system npm, not looking for it in the venv
    if sys.platform == 'win32':
        frontend_process = subprocess.Popen(
            "cmd /c npm run dev", 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    else:
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    
    os.chdir("..")
    
    # Print frontend output and look for the local URL
    frontend_url = None
    for line in frontend_process.stdout:
        print(f"[Frontend] {line.strip()}")
        
        # Try to extract the local URL
        if "Local:" in line and not frontend_url:
            parts = line.split("Local:")
            if len(parts) > 1:
                frontend_url = parts[1].strip()
                print(f"Frontend available at: {frontend_url}")
                
                # Open browser after a short delay
                threading.Timer(2.0, lambda: webbrowser.open(frontend_url)).start()

def cleanup():
    """Cleanup processes on exit"""
    print("\nShutting down...")
    
    if frontend_process:
        print("Terminating frontend process...")
        if sys.platform == 'win32':
            # On Windows, we need to use taskkill to terminate the process tree
            try:
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(frontend_process.pid)])
            except Exception as e:
                print(f"Error terminating frontend process: {e}")
        else:
            frontend_process.terminate()
    
    if backend_process:
        print("Terminating backend process...")
        backend_process.terminate()
    
    print("Shutdown complete.")

def signal_handler(sig, frame):
    """Handle interrupt signal"""
    cleanup()
    sys.exit(0)

def check_npm_installed():
    """Check if npm is available in the system (not in venv)"""
    try:
        # For Windows, use cmd to check npm version
        if sys.platform == 'win32':
            npm_version = subprocess.check_output("cmd /c npm --version", shell=True, text=True).strip()
        else:
            npm_version = subprocess.check_output(["npm", "--version"], text=True).strip()
        
        print(f"npm version: {npm_version}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: npm not found. Please install Node.js and npm.")
        return False

def verify_backend_requirements():
    """Verify that backend requirements are satisfied"""
    # Check for .env file with API keys
    if not os.path.exists(dotenv_path):
        print("Warning: No .env file found. LLM functionality may be limited.")
    
    # Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment. LLM functionality will be disabled.")
    
    # Check for database directory
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_store")
    if not os.path.exists(db_dir):
        print(f"Warning: Database directory not found at {db_dir}")
        print("Vector database functionality may be limited.")
    
    return True

if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Print welcome message
    print("="*50)
    print("Starting Agastya Web Application")
    print("="*50)
    
    # Verify requirements
    if not check_npm_installed():
        sys.exit(1)
    
    if not verify_backend_requirements():
        print("Some backend requirements are not met. Proceeding with limited functionality.")
    
    # Check if frontend dependencies are installed
    frontend_node_modules = os.path.join("frontend", "node_modules")
    if not os.path.exists(frontend_node_modules):
        print("Frontend dependencies not installed. Installing now...")
        os.chdir("frontend")
        if sys.platform == 'win32':
            subprocess.call("cmd /c npm install", shell=True)
        else:
            subprocess.call(["npm", "install"])
        os.chdir("..")
        print("Frontend dependencies installed.")
    
    # Create and start threads
    backend_thread = threading.Thread(target=run_backend)
    frontend_thread = threading.Thread(target=run_frontend)
    
    try:
        # Start the backend first
        backend_thread.start()
        
        # Give the backend a moment to start
        time.sleep(2)
        
        # Then start the frontend
        frontend_thread.start()
        
        # Print instructions
        print("\n"+"="*50)
        print("Agastya is starting up. You should see a browser window open shortly.")
        print("If not, check the URLs in the output above.")
        print("Press Ctrl+C to stop the application.")
        print("="*50+"\n")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cleanup() 