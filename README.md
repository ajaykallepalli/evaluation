# LLM API Evaluation Pipeline

A pipeline to evaluate and compare LLM API response times and responses for both direct question answering and RAG-based approaches.

## Requirements

Install the required Python package:
```bash
pip install -r requirements.txt
```

## Files

- `process_queries.py`: Baseline script for direct LLM question answering
- `process_queries_rag.py`: RAG-based script using embeddings and retrieval
- `test_queries.csv`: Input file containing queries to evaluate (CSV format)
- `llama3_baseline.csv`: Output file for baseline results
- `llama3_rag.csv`: Output file for RAG-based results
- `requirements.txt`: Python package dependencies

## Input Format

The `test_queries.csv` should have the following columns:
- document: Reference document for evaluation (used in RAG, unused in baseline)
- query: The question to be answered by the LLM
- desired_output: Expected response for evaluation
- observations: (optional) Any observations about the query
- other_notes: (optional) Additional notes

## Output Formats

### Baseline Output (`llama3_baseline.csv`)
- id: Sequential ID of the query
- timestamp: When the query was processed
- document: Reference document (unused in baseline)
- query: The full question sent to the LLM
- response: The actual response from the LLM
- response_time: Time taken for the API to respond (in seconds)
- desired_output: Expected response for evaluation
- curl_command: Complete curl command used (for reference/debugging)
- raw_response: Complete JSON response from the API

### RAG Output (`llama3_rag.csv`)
- id: Sequential ID of the query
- timestamp: When the query was processed
- document: Reference document used for retrieval
- query: The full question sent to the LLM
- embedding_time: Time taken for embedding generation
- embedding_response: Raw response from embedding API
- retrieval_time: Time taken for context retrieval
- retrieval_response: Raw response from retrieval API
- completion_time: Time taken for LLM completion
- completion_response: Raw response from LLM API
- total_time: Total processing time
- desired_output: Expected response for evaluation

## Usage

1. Ensure your queries are in `test_queries.csv`

2. Run baseline evaluation:
   ```bash
   python process_queries.py
   ```

3. Run RAG-based evaluation:
   ```bash
   python process_queries_rag.py
   ```

## Configuration

### Baseline Configuration
- Model ID: meta-llama/Llama-3.3-70B-Instruct
- Endpoint: http://localhost:9000/v1/chat/completions
- System prompt: "You are a helpful AI assistant. Please answer the following question to the best of your ability."

### RAG Configuration
- Model ID: meta-llama/Llama-3.3-70B-Instruct
- Embedding Endpoint: http://localhost:6006/embed
- Retrieval Endpoint: http://localhost:7000/v1/retrieval
- Completion Endpoint: http://localhost:9000/v1/chat/completions
- Graph Name: GRAPH
- System prompt: Includes retrieved context in the prompt 

## Word2Vec Homework Assignment

This repository includes a complete implementation of the Word2Vec algorithm for natural language processing.

### Features
- Word2Vec implementation using gensim
- Data preprocessing and exploration
- Experimentation with different hyperparameters (vector size, window size)
- t-SNE visualization of word embeddings
- Word similarity analysis
- Application: Sentiment analysis using Word2Vec embeddings
- Discussion of Word2Vec strengths and weaknesses

### Usage
1. Install requirements:
```
pip install -r requirements.txt
```

2. Run the script:
```
python word2vec_homework.py
```

The script will:
- Download Amazon reviews dataset
- Preprocess the text data
- Train multiple Word2Vec models with different hyperparameters
- Visualize word embeddings using t-SNE
- Perform word similarity analysis
- Implement sentiment analysis using Word2Vec embeddings
- Discuss strengths and weaknesses of Word2Vec

### Output
- Console output showing training progress and results
- t-SNE visualization saved as `word_embeddings_visualization.png`
- Sentiment analysis performance metrics 