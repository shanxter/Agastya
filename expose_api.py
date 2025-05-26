#!/usr/bin/env python
"""
This script runs ngrok to expose your local API server to the internet.
Run this script after starting your API server with api_server.py.

First, sign up for a free account at https://ngrok.com/
Then, get your authtoken from https://dashboard.ngrok.com/auth/your-authtoken

HOW TO USE:
1. Make sure your API server is running (python api_server.py)
2. Run this script with your authtoken as an argument:
   python expose_api.py 2aBcD3fGhIjKlMnOpQrStUvWxYz_1a2b3c4d5e6f7g8h9i
   
   NOTE: Replace the example authtoken above with your actual token from ngrok dashboard
"""

import os
import sys
import subprocess
import time
import webbrowser
from pyngrok import ngrok, conf

def main():
    # Check if ngrok is installed and set up
    if len(sys.argv) > 1:
        auth_token = sys.argv[1]
        conf.get_default().auth_token = auth_token
    else:
        print("No ngrok authtoken provided.")
        print("You can provide it as a command-line argument: python expose_api.py your_auth_token")
        print("Or set it up manually with: ngrok authtoken your_auth_token")
        print("\nTrying to continue without explicitly setting auth token...")
    
    # Set local API port (must match the port in api_server.py)
    api_port = 5000
    
    try:
        # Check if the API is running
        try:
            import requests
            response = requests.get(f"http://localhost:{api_port}/api/health")
            if response.status_code != 200:
                print(f"Warning: API server may not be running (got status code {response.status_code}).")
                choice = input("Do you want to continue anyway? (y/n): ")
                if choice.lower() != 'y':
                    return
        except Exception as e:
            print(f"Warning: Could not connect to API server: {e}")
            print(f"Make sure the API server is running on port {api_port} before continuing.")
            choice = input("Do you want to continue anyway? (y/n): ")
            if choice.lower() != 'y':
                return
        
        # Start ngrok tunnel
        print(f"Starting ngrok tunnel to http://localhost:{api_port}...")
        public_url = ngrok.connect(api_port, bind_tls=True).public_url
        
        print("\n" + "="*60)
        print(f"✨ Public URL: {public_url} ✨")
        print("Share this URL to allow others to access your application from anywhere!")
        print("="*60)
        
        # Instructions for frontend
        print("\nFor the frontend to work with this URL:")
        print("1. Update VITE_API_URL in your frontend/.env file to use this URL")
        print("   Example: VITE_API_URL=https://your-ngrok-url.ngrok.io/api")
        print("2. Restart your frontend development server")
        print("\nPress CTRL+C to stop the tunnel when you're done.")
        
        # Open browser automatically
        webbrowser.open(public_url)
        
        # Keep the tunnel running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down ngrok tunnel...")
    finally:
        # Clean up
        ngrok.kill()
        print("Ngrok tunnel closed.")

if __name__ == "__main__":
    main() 