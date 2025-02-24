import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1️⃣ Load Open-Source LLM (Free Hugging Face Model)
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

def call_llm(prompt):
    payload = {"inputs": prompt}
    try:
#        logging.info("Calling LLM API with prompt: %s", prompt)
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        response_json = response.json()

        if isinstance(response_json, list) and "generated_text" in response_json[0]:
            return response_json[0]["generated_text"]
        else:
            logging.warning("Unexpected API Response: %s", response_json)
            return None
    except requests.exceptions.RequestException as err:
        logging.error("API Error: %s", err)
    return None

# 2️⃣ Load Embedding Model (Hugging Face BGE)
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en")
model = AutoModel.from_pretrained("BAAI/bge-small-en")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    
    embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    logging.info("Generated embedding for text: %s", text)
    return embedding

# 3️⃣ Create FAISS Index for Log Embeddings
EMBEDDING_DIM = 384  # Adjusted for BGE model
faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
log_db = []

def store_log(log_text):
#    logging.info("Storing log: %s", log_text)
    embedding = get_embedding(log_text)
    faiss_index.add(np.array([embedding]))
    log_db.append(log_text)

def search_similar_logs(query_text):
    query_embedding = get_embedding(query_text)
    D, I = faiss_index.search(np.array([query_embedding]), k=3)
    results = [log_db[i] for i in I[0] if i != -1 and i < len(log_db)]
    logging.info("Similar logs found for query '%s': %s", query_text, results)
    return results

# 4️⃣ Sample Log Entries
log_entries = [
"[2025-02-21 08:00:01] INFO  WebServer started on port 8080."
"[2025-02-21 08:00:05] DEBUG Database connection established."
"[2025-02-21 08:01:10] INFO  User 'admin' logged in successfully."
"[2025-02-21 08:01:15] DEBUG Fetching user preferences from DB."
"[2025-02-21 08:02:20] WARN  Slow query detected: SELECT * FROM users WHERE last_login > 30 days."
"[2025-02-21 08:03:05] INFO  Background job 'cache_refresh' completed in 2.1s."
"[2025-02-21 08:04:12] INFO  Email notification sent to user@example.com."
"[2025-02-21 08:05:30] DEBUG Request received: GET /api/v1/status."
"[2025-02-21 08:06:40] INFO  File uploaded: report_2025.pdf (2.5MB)."
"[2025-02-21 08:07:55] WARN  High memory usage detected (85% used)."
"[2025-02-21 08:08:20] INFO  Cache cleared successfully."
"[2025-02-21 08:09:10] DEBUG Fetching latest transactions from DB."
"[2025-02-21 08:10:25] INFO  User 'john_doe' logged out."
"[2025-02-21 08:12:10] ERROR Failed to connect to Redis: Connection refused."
"[2025-02-21 08:13:05] INFO  Retrying Redis connection..."
"[2025-02-21 08:14:00] INFO  Redis connection established successfully."
"[2025-02-21 08:15:30] WARN  CPU usage exceeded 90% threshold."
"[2025-02-21 08:16:45] INFO  Scheduled task 'backup' started."
"[2025-02-21 08:18:00] DEBUG Backup progress: 50% completed."
"[2025-02-21 08:19:15] INFO  Backup completed successfully."
"[2025-02-21 08:20:40] INFO  Service 'authenticator' restarted after 5 minutes of inactivity."
"[2025-02-21 08:22:10] ERROR FileNotFoundException: config.yaml not found."
"[2025-02-21 08:23:45] INFO  Default config loaded as fallback."
"[2025-02-21 08:25:20] DEBUG Running database maintenance."
"[2025-02-21 08:27:05] INFO  Database maintenance completed."
"[2025-02-21 08:30:10] WARN  Disk space running low (5GB left)."
"[2025-02-21 08:32:50] INFO  Purged old logs older than 90 days."
"[2025-02-21 08:35:30] ERROR API request failed: 500 Internal Server Error."
"[2025-02-21 08:38:15] INFO  Retrying API request..."
"[2025-02-21 08:40:00] INFO  API request succeeded."
"[2025-02-21 08:42:20] DEBUG Fetching active user sessions."
"[2025-02-21 08:45:10] INFO  Load balancer reconfigured for traffic spike."
"[2025-02-21 08:48:55] WARN  High response time detected on endpoint /api/v1/orders."
"[2025-02-21 08:50:30] INFO  Started new worker thread to handle queue backlog."
"[2025-02-21 08:52:15] INFO  SSL certificate renewed successfully."
"[2025-02-21 08:55:10] ERROR OutOfMemoryError: Heap space exceeded."
"[2025-02-21 08:58:00] INFO  Restarting service due to memory pressure."
"[2025-02-21 09:00:30] DEBUG Checking for software updates."
"[2025-02-21 09:02:10] INFO  No updates available."
"[2025-02-21 09:05:15] WARN  Unusual login attempt detected from IP 192.168.1.5."
"[2025-02-21 09:07:50] INFO  Security alert triggered for suspicious activity."
"[2025-02-21 09:10:00] INFO  Admin review requested for flagged account."
"[2025-02-21 09:12:45] DEBUG Processing transaction ID #897654."
"[2025-02-21 09:15:20] INFO  Transaction completed successfully."
"[2025-02-21 09:18:30] WARN  Service latency increasing beyond acceptable range."
"[2025-02-21 09:20:10] INFO  Increased resource allocation to stabilize performance."
]

for log_entry in log_entries:
    parsed_log = call_llm(f"Extract structured data from log: {log_entry}")
    if parsed_log:
        store_log(parsed_log)

# 5️⃣ Detect Anomalies & Perform Root Cause Analysis
new_log = [
 "[2025-02-22 10:05:10] INFO  ServiceX started on port 9000."
 "[2025-02-22 10:06:30] DEBUG Initializing cache system."
 "[2025-02-22 10:07:50] INFO  New client connection established."
 "[2025-02-22 10:09:15] DEBUG Processing user authentication request."
 "[2025-02-22 10:10:25] WARN  Response time exceeding 500ms."
 "[2025-02-22 10:12:10] INFO  Flushing temporary files."
 "[2025-02-22 10:14:50] ERROR Segmentation Fault: Attempted to access invalid memory address."
 "[2025-02-22 10:15:10] INFO  Restarting ServiceX after crash..."
 "[2025-02-22 10:16:30] DEBUG Restoring session state."
 "[2025-02-22 10:18:00] WARN  Recovery process took longer than expected (15s)."
 "[2025-02-22 10:20:05] ERROR ServiceX encountered another Segmentation Fault."
 "[2025-02-22 10:22:15] INFO  Disabling faulty module and logging incident report."
]

logging.info("Analyzing new log entry: %s", new_log)
similar_logs = search_similar_logs(new_log)

if not similar_logs:
    logging.warning("🚨 Anomaly Detected!")
else:
    logging.info("✅ Similar logs found: %s", similar_logs)

rca_prompt = f"Analyze root cause for: {new_log}\nSimilar past incidents: {similar_logs}"
root_cause = call_llm(rca_prompt)
if root_cause:
    logging.info("🔍 Root Cause Analysis: %s", root_cause)
