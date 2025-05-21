"""
Test import resolution for critical packages in the project.
Run this script to verify all packages are properly installed.
"""
import sys
import importlib.util
import os

def check_import(module_name):
    """Check if a module can be imported and is in the Python path."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name} from {module.__file__}")
        try:
            print(f"   Version: {module.__version__}")
        except AttributeError:
            print("   No version information available")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    """Run import checks for all critical modules."""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print("-" * 60)
    
    critical_modules = [
        "langchain_core",
        "langchain_openai",
        "langchain_community",
        "langchain_text_splitters",
        "openai",
        "chromadb",
        "dotenv",
        "tiktoken",
        "requests"
    ]
    
    # Check each module
    results = []
    for module in critical_modules:
        results.append(check_import(module))
    
    # Summary
    print("-" * 60)
    print(f"Summary: {sum(results)}/{len(results)} modules successfully imported")
    if all(results):
        print("✅ All imports resolved successfully")
    else:
        print("❌ Some imports failed to resolve")

if __name__ == "__main__":
    main() 