import os
import requests # Added for making HTTP requests
from flask import Flask, request, jsonify
from prometheus_client import generate_latest, Counter, Histogram, Gauge
import time
import pandas as pd
# Removed pathlib import as the direct model loading is removed

app = Flask(__name__)

# --- Configuration for MLflow Model Serve Endpoint ---
# This is the endpoint where 'mlflow model serve' will be running.
# The reviewer suggested 5005, let's use that.
MLFLOW_MODEL_SERVE_URL = os.getenv('MLFLOW_MODEL_SERVE_URL', 'http://127.0.0.1:5005/predict')

# --- Prometheus Metrics ---
PREDICTIONS_TOTAL = Counter(
    'ml_model_predictions_total',
    'Total number of predictions made by the ML model.'
)

PREDICTION_DURATION_SECONDS = Histogram(
    'ml_model_prediction_duration_seconds',
    'Histogram of prediction duration in seconds.',
    buckets=(.001, .005, .01, .025, .05, .075, .1, .25, .5, 1.0, 2.5, 5.0, 10.0, float('inf'))
)

# This gauge will now represent the *reachability* of the MLflow model serve endpoint
MLFLOW_SERVE_STATUS = Gauge(
    'mlflow_model_serve_status',
    'Gauge indicating if the MLflow model serve endpoint is reachable (1 for reachable, 0 for unreachable).'
)

# NOTE: MODEL_LOAD_SUCCESS gauge from direct loading is removed/changed
# as inference.py no longer directly loads the model.

# Function to check if the MLflow serve endpoint is up
def check_mlflow_serve_health():
    try:
        response = requests.get(MLFLOW_MODEL_SERVE_URL.replace('/predict', '/ping'), timeout=1) # Ping endpoint
        if response.status_code == 200:
            MLFLOW_SERVE_STATUS.set(1)
            return True
        MLFLOW_SERVE_STATUS.set(0)
        return False
    except requests.exceptions.RequestException:
        MLFLOW_SERVE_STATUS.set(0)
        return False

@app.route('/predict', methods=['POST'])
def predict():
    # Model is served externally, check if endpoint is reachable
    if not check_mlflow_serve_health():
        return jsonify({"error": "MLflow model serve endpoint is unreachable or unhealthy."}), 503

    start_time = time.time()
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON input"}), 400

        # Send request to MLflow model serve endpoint
        headers = {"Content-Type": "application/json"}
        response = requests.post(MLFLOW_MODEL_SERVE_URL, headers=headers, json=data)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        prediction_result = response.json()
        
        PREDICTIONS_TOTAL.inc() # Increment after successful external prediction

        end_time = time.time()
        PREDICTION_DURATION_SECONDS.observe(end_time - start_time)

        # Assuming prediction_result is a list or single value
        return jsonify(prediction_result)
    except requests.exceptions.HTTPError as e:
        print(f"Prediction error from MLflow serve: HTTPError - {e.response.status_code} {e.response.text}")
        return jsonify({"error": f"Error from model serve: {e.response.text}"}), e.response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Prediction error connecting to MLflow serve: {e}")
        return jsonify({"error": "Failed to connect to MLflow model serve endpoint."}), 503
    except Exception as e:
        print(f"Unexpected prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    # Ensure exporter status is updated on metrics scrape
    check_mlflow_serve_health() 
    return generate_latest(), 200

@app.route('/health')
def health_check():
    if check_mlflow_serve_health():
        return "MLflow model serve endpoint is reachable", 200
    return "MLflow model serve endpoint is unreachable", 503

if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 5001)) # Running on 5001
    print(f"Starting inference.py proxy API on {host}:{port}")
    app.run(host=host, port=port, debug=False)