import datetime
import json
import os

import numpy as np

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.int64) or \
            isinstance(obj, np.int32) or \
            isinstance(obj, np.int16) or \
            isinstance(obj, np.int8):
            return int(obj)

        if isinstance(obj, np.float128) or \
            isinstance(obj, np.float96) or \
            isinstance(obj, np.float64) or \
            isinstance(obj, np.float32) or \
            isinstance(obj, np.float16):
            return float(obj)

        return json.JSONEncoder.default(self, obj)

def store_logs(logs):
    folder_path = "logs"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"{timestamp}.json")

    os.makedirs(folder_path, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, cls=CustomEncoder)

    print(f"Log was saved to {file_path}")

def load_log(log_filename: str):
    folder_path = "logs"
    file_path = os.path.join(folder_path, log_filename)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

