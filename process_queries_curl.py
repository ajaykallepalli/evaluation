#!/usr/bin/env python3
import csv
import json
import subprocess
from datetime import datetime
import time

# Configuration
INPUT_FILE = "test_queries.csv"
OUTPUT_FILE = "llama3_chatqna_test.csv"
ENDPOINT = "http://localhost:8888/v1/chatqna"

def process_query(query):
    """Process a single query using curl command"""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    # Convert payload to JSON string
    json_payload = json.dumps(payload)
    
    # Construct curl command
    curl_cmd = [
        "curl", "-X", "POST", ENDPOINT,
        "-H", "Content-Type: application/json",
        "-d", json_payload
    ]
    
    start_time = time.time()
    
    try:
        # Run curl command and capture output
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.returncode != 0:
            print(f"Error processing query: {result.stderr}")
            return None, processing_time, result.stderr
        
        # Parse and clean the response
        response_lines = result.stdout.split('\n')
        cleaned_response = []
        
        for line in response_lines:
            if line.startswith("data: b'") and not line.endswith("data: b''"):
                # Extract the content between the quotes
                content = line[8:-1]  # Remove "data: b'" and "'"
                cleaned_response.append(content)
        
        # Join all parts of the response
        full_response = ''.join(cleaned_response)
        return full_response, processing_time, None
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Exception processing query: {str(e)}")
        return None, processing_time, str(e)

def main():
    # Create output file with headers
    fieldnames = [
        'id', 'timestamp', 'document', 'query',
        'processing_time', 'error', 'response',
        'desired_output'
    ]
    
    # Read test queries to get desired outputs
    test_queries = {}
    with open(INPUT_FILE, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            test_queries[row['query']] = row['desired_output']
    
    with open(OUTPUT_FILE, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each query
        with open(INPUT_FILE, 'r') as infile:
            reader = csv.DictReader(infile)
            for idx, row in enumerate(reader, 1):
                print(f"\nProcessing query {idx}: {row['query']}")
                
                # Process query
                response, processing_time, error = process_query(row['query'])
                
                # Write results
                writer.writerow({
                    'id': idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'document': row['document'],
                    'query': row['query'],
                    'processing_time': f"{processing_time:.3f}",
                    'error': error if error else '',
                    'response': response if response else '',
                    'desired_output': test_queries.get(row['query'], '')
                })
                
                print(f"Total processing time: {processing_time:.3f} seconds")
                print("-" * 40)
    
    print(f"\nProcessing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 