#!/bin/bash
# Set model ID
export LLM_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
ENDPOINT="http://localhost:9000/v1/chat/completions"
INPUT_FILE="test_queries.csv"
OUTPUT_FILE="llama3_baseline.csv"

# Create output file with headers
echo "id,timestamp,document,query,response_time,desired_output,curl_command" > $OUTPUT_FILE

# Process each line in the CSV file
ID=1
FIRST_LINE=true

while IFS=, read -r document query desired_output observations other_notes; do
    # Skip header
    if [ "$FIRST_LINE" = true ]; then
        FIRST_LINE=false
        continue
    fi
    # Remove quotes from document and query
    document="${document//\"/}"
    query="${query//\"/}"
    desired_output="${desired_output//\"/}"
    echo "Processing query $ID: $query"
    # Create JSON payload
    PAYLOAD="{\"model\": \"$LLM_MODEL_ID\", \"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful AI assistant. Please answer the following question to the best of your ability.\"}, {\"role\": \"user\", \"content\": \"$query\"}]}"
    # Format curl command for output
    CURL_CMD="curl -s $ENDPOINT -H \"Content-Type: application/json\" -d '$PAYLOAD'"
    # Run the curl command and time it
    echo "Running curl command..."
    START_TIME=$(date +%s.%N)
    curl -s $ENDPOINT -H "Content-Type: application/json" -d "$PAYLOAD" > /dev/null
    END_TIME=$(date +%s.%N)
    # Calculate response time
    RESPONSE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    echo "Response time: $RESPONSE_TIME seconds"
    # Save results to CSV
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    ESCAPED_CURL_CMD=$(echo "$CURL_CMD" | sed 's/,/\\,/g')
    echo "$ID,$TIMESTAMP,$document,$query,$RESPONSE_TIME,$desired_output,$ESCAPED_CURL_CMD" >> $OUTPUT_FILE
    # Increment ID
    ID=$((ID + 1))
    echo "----------------------------------------"
done < $INPUT_FILE

echo "Processing complete. Results saved to $OUTPUT_FILE"
