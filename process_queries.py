#!/usr/bin/env python3
import csv
import json
import requests
from datetime import datetime
import time

# Configuration
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
ENDPOINT = "http://localhost:9000/v1/chat/completions"
INPUT_FILE = "test_queries.csv"
OUTPUT_FILE = "llama3_baseline.csv"

def process_query(query, model_id):
    """Send query to LLM and return response with timing"""
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Please answer the following question to the best of your ability."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    start_time = time.time()
    response = requests.post(
        ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload
    )
    end_time = time.time()
    response_time = end_time - start_time
    
    response_json = response.json()
    response_content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    return {
        'response': response_content,
        'response_time': response_time,
        'raw_response': response.text,
        'curl_command': f"curl -s {ENDPOINT} -H 'Content-Type: application/json' -d '{json.dumps(payload)}'"
    }

def main():
    # Create output file with headers
    fieldnames = [
        'id', 'timestamp', 'document', 'query', 'response',
        'response_time', 'desired_output', 'curl_command', 'raw_response'
    ]
    
    with open(OUTPUT_FILE, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each query
        with open(INPUT_FILE, 'r') as infile:
            reader = csv.DictReader(infile)
            for idx, row in enumerate(reader, 1):
                print(f"Processing query {idx}: {row['query']}")
                
                result = process_query(row['query'], MODEL_ID)
                
                # Write results
                writer.writerow({
                    'id': idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'document': row['document'],
                    'query': row['query'],
                    'response': result['response'],
                    'response_time': f"{result['response_time']:.3f}",
                    'desired_output': row['desired_output'],
                    'curl_command': result['curl_command'],
                    'raw_response': result['raw_response']
                })
                
                print(f"Response time: {result['response_time']:.3f} seconds")
                print("-" * 40)
    
    print(f"Processing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 