import os
import shutil
import yaml
import time
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import RAGEngine
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationTest")

TEST_DIR = "tests"
DATA_DIR = os.path.join(TEST_DIR, "data")
DB_DIR = os.path.join(TEST_DIR, "chroma_db")
CONFIG_PATH = os.path.join(TEST_DIR, "test_config.yaml")

def setup_test_env():
    """Create test directories and config"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Clean DB dir if exists
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    
    # Create test config
    config = {
        'rag': {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'top_k': 3,
            'context_count': 3,
            'search_type': 'hybrid_weighted',
            'use_rerank': False, # Disable for speed in tests
        },
        'vector_db': {
            'path': DB_DIR,
            'collection_name': 'test_collection',
            'hnsw': {
                'M': 16,
                'ef_construction': 100,
                'ef': 30
            }
        },
        'llm': {
            'model_name': 'qwen2:7b',
            'temperature': 0.1,
            'max_tokens': 200
        }
    }
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)
        
    logger.info("Test environment setup complete.")

def create_test_files():
    """Generate test files"""
    # 1. Normal PDF
    pdf_path = os.path.join(DATA_DIR, "normal.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "The capital of France is Paris.")
    c.drawString(100, 730, "The capital of Japan is Tokyo.")
    c.save()
    
    # 2. Empty PDF
    empty_path = os.path.join(DATA_DIR, "empty.pdf")
    c = canvas.Canvas(empty_path, pagesize=letter)
    c.save() # No text
    
    # 3. Corrupted PDF
    corrupted_path = os.path.join(DATA_DIR, "corrupted.pdf")
    with open(corrupted_path, 'w') as f:
        f.write("This is not a PDF content.")
        
    # 4. Special Char Filename
    special_path = os.path.join(DATA_DIR, "special_!@#.txt")
    with open(special_path, 'w') as f:
        f.write("This file has special characters in its name.")

    # 5. Large Text File
    large_path = os.path.join(DATA_DIR, "large.txt")
    with open(large_path, 'w') as f:
        f.write("Repeat this sentence. " * 5000)

    logger.info("Test files created.")
    return {
        'normal': pdf_path,
        'empty': empty_path,
        'corrupted': corrupted_path,
        'special': special_path,
        'large': large_path
    }

def run_tests():
    setup_test_env()
    files = create_test_files()
    
    engine = RAGEngine(config_path=CONFIG_PATH)
    
    results = {}
    
    # Test 1: Normal Indexing & Query
    logger.info("--- Test 1: Normal Indexing & Query ---")
    res = engine.load_and_index_file(files['normal'])
    if res['success']:
        q_res = engine.query("What is the capital of France?")
        if "Paris" in q_res['answer']:
            results['test_1'] = "PASS"
        else:
            results['test_1'] = f"FAIL (Answer: {q_res['answer']})"
    else:
        results['test_1'] = f"FAIL (Indexing: {res['message']})"
        
    # Test 2: Empty PDF
    logger.info("--- Test 2: Empty PDF ---")
    res = engine.load_and_index_file(files['empty'])
    # Expect success=False or warning, but NO CRASH
    # Based on loaders.py, empty file might return empty docs, and indexer might fail gracefully
    if not res['success'] or res['num_documents'] == 0:
        results['test_2'] = "PASS"
    else:
        results['test_2'] = f"FAIL (Unexpected success: {res})"

    # Test 3: Corrupted PDF
    logger.info("--- Test 3: Corrupted PDF ---")
    res = engine.load_and_index_file(files['corrupted'])
    if not res['success']:
        results['test_3'] = "PASS"
    else:
        results['test_3'] = f"FAIL (Unexpected success: {res})"
        
    # Test 4: Special Char Filename
    logger.info("--- Test 4: Special Char Filename ---")
    res = engine.load_and_index_file(files['special'])
    if res['success']:
        results['test_4'] = "PASS"
    else:
        results['test_4'] = f"FAIL ({res['message']})"

    # Test 5: Large File
    logger.info("--- Test 5: Large File ---")
    res = engine.load_and_index_file(files['large'])
    if res['success']:
        results['test_5'] = "PASS"
    else:
        results['test_5'] = f"FAIL ({res['message']})"

    # Test 6: Incremental Indexing
    logger.info("--- Test 6: Incremental Indexing ---")
    # Modify normal.pdf
    c = canvas.Canvas(files['normal'], pagesize=letter)
    c.drawString(100, 750, "The capital of Germany is Berlin.") # Changed content
    c.save()
    
    # Re-index
    res = engine.load_and_index_file(files['normal'])
    # Should detect change and re-index
    if res['success']:
        # Query for new content
        q_res = engine.query("What is the capital of Germany?")
        if "Berlin" in q_res['answer']:
            results['test_6'] = "PASS"
        else:
            results['test_6'] = f"FAIL (New content not found: {q_res['answer']})"
    else:
        results['test_6'] = f"FAIL (Re-indexing failed: {res['message']})"

    # Test 7: Settings Change (Runtime)
    logger.info("--- Test 7: Settings Change ---")
    # Change context_count to 1
    engine.config['rag']['context_count'] = 1
    q_res = engine.query("What is the capital of Germany?")
    if len(q_res['search_results']) == 1:
        results['test_7'] = "PASS"
    else:
        results['test_7'] = f"FAIL (Expected 1 result, got {len(q_res['search_results'])})"

    # Summary
    print("\n=== Integration Test Results ===")
    for test, result in results.items():
        print(f"{test}: {result}")
        
    if all(r == "PASS" for r in results.values()):
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    run_tests()
