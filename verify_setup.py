import os
import sys

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ Import successful: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {module_name} - {e}")
        return False

def check_file(path):
    if os.path.exists(path):
        print(f"✅ File/Dir exists: {path}")
        return True
    else:
        print(f"❌ Missing: {path}")
        return False

def main():
    print("Starting Environment Verification...")
    
    # Check Imports
    modules = [
        "langchain",
        "streamlit",
        "chromadb",
        "sentence_transformers",
        "pypdf",
        "rank_bm25",
        "markdown"
    ]
    
    all_imports_ok = all(check_import(m) for m in modules)
    
    # Check Project Files
    files = [
        "config.yaml",
        "app.py",
        "rag_engine.py",
        "loaders.py",
        "indexer.py",
        "retriever.py",
        "llm_client.py"
    ]
    
    all_files_ok = all(check_file(f) for f in files)
    
    if all_imports_ok and all_files_ok:
        print("\n🎉 All checks passed! Environment is ready.")
    else:
        print("\n⚠️ Some checks failed. Please review above.")

if __name__ == "__main__":
    main()
