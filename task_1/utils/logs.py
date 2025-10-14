"""Utility functions for storing and loading logs."""
import datetime
import json
import os

import numpy as np

class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, (np.int64, np.int32, np.int16, np.int8)):
            return int(o)

        if isinstance(o, (np.float64, np.float32, np.float16)):
            return float(o)

        return json.JSONEncoder.default(self, o)

def store_logs(logs):
    """Store logs in a JSON file with a timestamped filename."""
    folder_path = "logs"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"{timestamp}.json")

    os.makedirs(folder_path, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, cls=CustomEncoder)

    print(f"Log was saved to {file_path}")

def load_log(log_filename: str):
    """Load logs from a specified JSON file."""
    folder_path = "logs"
    file_path = os.path.join(folder_path, log_filename)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
