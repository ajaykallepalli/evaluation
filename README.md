# LLM API Evaluation Pipeline

A simple pipeline to evaluate LLM API response times and store results.

## Files

- `process_all_queries_fix.sh`: Main script that processes queries and measures response times
- `test_queries.csv`: Input file containing queries to evaluate (CSV format)
- `llama3_baseline.csv`: Output file with evaluation results

## Input Format

The `test_queries.csv` should have the following columns:
- document: Context document for the query
- query: The actual query to send
- desired_output: Expected response
- observations: (optional) Any observations about the query
- other_notes: (optional) Additional notes

## Output Format

The `llama3_baseline.csv` contains:
- id: Sequential ID of the query
- timestamp: When the query was processed
- document: Context document used
- query: The query that was sent
- response_time: Time taken for the API to respond (in seconds)
- desired_output: Expected response
- curl_command: The actual curl command used (for reference)

## Usage

1. Ensure your queries are in `test_queries.csv`
2. Make the script executable if needed:
   ```bash
   chmod +x process_all_queries_fix.sh
   ```
3. Run the evaluation:
   ```bash
   ./process_all_queries_fix.sh
   ```

## Configuration

The script uses these default settings (modify in script if needed):
- Model ID: meta-llama/Llama-3.3-70B-Instruct
- Endpoint: http://localhost:9000/v1/chat/completions
- Input file: test_queries.csv
- Output file: llama3_baseline.csv 