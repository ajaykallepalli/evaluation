#!/usr/bin/env python3
import csv
import json
import requests
from datetime import datetime
import time

# Configuration
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
EMBEDDING_ENDPOINT = "http://localhost:6006/embed"
RETRIEVAL_ENDPOINT = "http://localhost:7000/v1/retrieval"
COMPLETION_ENDPOINT = "http://localhost:9000/v1/chat/completions"
INPUT_FILE = "test_queries.csv"
OUTPUT_FILE = "llama3_immunization_v2_rag_node.csv"
GRAPH_NAME = "GRAPH"
SEARCH_START = "node"

def get_embeddings(query):
    """Get embeddings for the query"""
    start_time = time.time()
    response = requests.post(
        EMBEDDING_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json={"inputs": query}
    )
    end_time = time.time()
    embedding_time = end_time - start_time
    
    if response.status_code != 200:
        print(f"Error getting embeddings: {response.text}")
        return None, embedding_time, response.text
    
    # The embedding response is a list of floats
    embedding = response.json()[0]  # Get first (and only) embedding
    return embedding, embedding_time, response.text

def get_retrieval(query, embedding):
    """Get relevant context using retrieval"""
    start_time = time.time()
    response = requests.post(
        RETRIEVAL_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json={
            "input": query,
            "embedding": embedding,
            "graph_name": GRAPH_NAME,
            "search_start": SEARCH_START
        }
    )
    end_time = time.time()
    retrieval_time = end_time - start_time
    
    if response.status_code != 200:
        print(f"Error in retrieval: {response.text}")
        return None, retrieval_time, response.text
    
    # Extract the retrieved text passages
    result = response.json()
    retrieved_texts = [doc["text"] for doc in result.get("retrieved_docs", [])]
    context = "\n\n".join(retrieved_texts)
        
    return context, retrieval_time, response.text

def get_completion(query, context):
    """Get LLM completion with context"""
    system_prompt = (
        "### You are a helpful, respectful and honest assistant to help the user with questions. "
        "Please refer to the search results obtained from the local knowledge base. "
        "But be careful to not incorporate the information that you think is not relevant to the question. "
        "If you don't know the answer to a question, please don't share false information.\n"
        f"### Search results: {context}\n"
        f"### Question: {query}\n"
        "### Answer:"
    )
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    start_time = time.time()
    response = requests.post(
        COMPLETION_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json=payload
    )
    end_time = time.time()
    completion_time = end_time - start_time
    
    if response.status_code != 200:
        print(f"Error in completion: {response.text}")
        return None, completion_time, response.text
        
    response_json = response.json()
    response_content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    return response_content, completion_time, response.text

def main():
    # Create output file with headers
    fieldnames = [
        'id', 'timestamp', 'document', 'query',
        'embedding_time', 'embedding_response',
        'retrieval_time', 'retrieval_response', 'retrieved_context',
        'completion_time', 'completion_response', 'llm_response',
        'total_time', 'desired_output'
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
                total_start_time = time.time()
                
                # Step 1: Get embeddings
                print("Getting embeddings...")
                embeddings, embedding_time, embedding_raw = get_embeddings(row['query'])
                if embeddings is None:
                    print("Skipping query due to embedding error")
                    continue
                
                # Step 2: Get retrieval results
                print("Getting retrieval results...")
                context, retrieval_time, retrieval_raw = get_retrieval(
                    row['query'],
                    embeddings
                )
                if context is None:
                    print("Skipping query due to retrieval error")
                    continue
                
                # Step 3: Get completion
                print("Getting completion...")
                completion, completion_time, completion_raw = get_completion(
                    row['query'],
                    context
                )
                if completion is None:
                    print("Skipping query due to completion error")
                    continue
                
                total_time = time.time() - total_start_time
                
                # Write results
                writer.writerow({
                    'id': idx,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'document': row['document'],
                    'query': row['query'],
                    'embedding_time': f"{embedding_time:.3f}",
                    'embedding_response': embedding_raw,
                    'retrieval_time': f"{retrieval_time:.3f}",
                    'retrieval_response': retrieval_raw,
                    'retrieved_context': context,
                    'completion_time': f"{completion_time:.3f}",
                    'completion_response': completion_raw,
                    'llm_response': completion,
                    'total_time': f"{total_time:.3f}",
                    'desired_output': test_queries.get(row['query'], '')
                })
                
                print(f"Total processing time: {total_time:.3f} seconds")
                print("-" * 40)
    
    print(f"\nProcessing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 