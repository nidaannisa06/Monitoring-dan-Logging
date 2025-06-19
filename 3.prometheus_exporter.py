import os
import time
import requests # Added for making HTTP requests
import mlflow.pyfunc # Kept for potential MLflow utility functions, though not loading model directly
from prometheus_client import start_http_server, Gauge
from mlflow.exceptions import MlflowException
import pandas as pd # Needed for sample data for requests
import json # Ensure json is imported
from pathlib import Path

# --- Configuration for MLflow Model Serve Endpoint ---
# FIXED: Changed from /predict to /invocations (MLflow's correct endpoint)
MLFLOW_MODEL_SERVE_URL = os.getenv('MLFLOW_MODEL_SERVE_URL', 'http://127.0.0.1:5005/invocations')

DUMMY_INPUT_JSON_PATH = Path("dummy_input.json")
# --- Prometheus Metrics ---
# Renamed from PREDICTION_GAUGE for consistency with inference.py's prediction counter
LAST_PREDICTION_VALUE = Gauge('ml_model_last_prediction_value', 'Last predicted house price value from MLflow serve.')
LATENCY_EXPORTER = Gauge('ml_exporter_prediction_latency_ms', 'Inference latency observed by the exporter.')
MLFLOW_SERVE_STATUS_EXPORTER = Gauge('mlflow_model_serve_status_exporter', 'Gauge indicating if the MLflow model serve endpoint is reachable by exporter.')


def get_sample_prediction_from_api():
    """Generates a sample prediction by calling the MLflow model serve endpoint."""
    sample_data = []
    
    try:
        if DUMMY_INPUT_JSON_PATH.exists():
            with open(DUMMY_INPUT_JSON_PATH, "r") as f:
                sample_data = json.load(f)
            print(f"Loaded dummy_input.json from {DUMMY_INPUT_JSON_PATH} for sample prediction.")
        else:
            print(f"Error: dummy_input.json not found at {DUMMY_INPUT_JSON_PATH}. Cannot generate sample prediction.")
            # Fallback to a very minimal hardcoded sample if file is genuinely missing
            # This hardcoded sample WILL LIKELY FAIL YOUR MODEL due to missing features.
            # It's better to ensure the file exists.
            sample_data = [{
                "total_sqft": 1000.0, "bath": 2.0, "bhk": 3.0, "price_per_sqft": 5000.0,
                # ... (add a few other critical common features if you have them, otherwise this minimal sample won't work)
                # This should ideally be a valid, albeit basic, input.
            }]
            return None, None # Indicate failure if file not found and hardcoded is incomplete

    except json.JSONDecodeError:
        print(f"Error: dummy_input.json at {DUMMY_INPUT_JSON_PATH} is not valid JSON.")
        return None, None
    except Exception as e:
        print(f"Unexpected error loading dummy_input.json: {e}")
        return None, None

    start_time = time.time()
    try:
        headers = {"Content-Type": "application/json"}
        # FIXED: Using the correct MLflow input format
        # MLflow expects data in "inputs" key for most models
        payload = {"inputs": sample_data}
        
        response = requests.post(MLFLOW_MODEL_SERVE_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        prediction_value = response.json()
        latency_ms = (time.time() - start_time) * 1000
        
        MLFLOW_SERVE_STATUS_EXPORTER.set(1)
        # Handle different response formats from MLflow
        if isinstance(prediction_value, list):
            return prediction_value[0], latency_ms
        elif isinstance(prediction_value, dict) and 'predictions' in prediction_value:
            return prediction_value['predictions'][0], latency_ms
        else:
            # If it's a single value
            return float(prediction_value), latency_ms
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling MLflow serve from exporter: {e}")
        print(f"Response status code: {getattr(e.response, 'status_code', 'N/A')}")
        print(f"Response text: {getattr(e.response, 'text', 'N/A')}")
        MLFLOW_SERVE_STATUS_EXPORTER.set(0)
        return None, None # Indicate failure
    except Exception as e:
        print(f"Unexpected prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus exporter started on port 8000")

    while True:
        prediction, latency = get_sample_prediction_from_api()
        if prediction is not None:
            LAST_PREDICTION_VALUE.set(prediction)
            LATENCY_EXPORTER.set(latency)
            print(f"Exporter: Sample prediction: {prediction:.2f}, Latency: {latency:.2f}ms")
        else:
            print("Exporter: Failed to get sample prediction from API.")
        time.sleep(15) # Scrape interval for Prometheus