import os
import sys
from rag.ingest import ingest_file

def run_tests():
    # Folder to store temporary test files
    test_dir = "data/test_files"
    os.makedirs(test_dir, exist_ok=True)

    # Test cases: (file_name, content)
    test_files = [
        ("test_normal.txt", "This is a regular text file with standard formatting."),
        ("ucp_invoice", "Invoice Number: 98765\nTotal: $150\nStatus: Paid"),
        ("file_no_extension", "This file has no extension at all but contains valid plain text."),
    ]

    print("Running Ingestion Tests...\n")
    passed = 0
    failed = 0

    for filename, content in test_files:
        file_path = os.path.join(test_dir, filename)
        
        # Write test content to disk
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Test the ingest function
        try:
            chunks = ingest_file(file_path)
            print(f" SUCCESS: '{filename}' ingested ({chunks} chunks created).")
            passed += 1
        except Exception as e:
            print(f" FAILED: '{filename}' failed to ingest. Error: {e}")
            failed += 1

        # Delete temporary test file
        if os.path.exists(file_path):
            os.remove(file_path)

    # Remove temporary test folder
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

    print(f"\nResults: {passed} passed, {failed} failed.")

if __name__ == "__main__":
    run_tests()