"""Download and prepare the Animals10 dataset from Kaggle."""
import os
from pathlib import Path
import shutil

import kagglehub

# Download latest version
path = kagglehub.dataset_download("alessiocorrado99/animals10")

print("Cached dataset files at:", path)

BASE_DIR = Path(__file__).resolve().parent

target_dir = Path(BASE_DIR / "../data/animals")
target_dir.mkdir(parents=True, exist_ok=True)

# Copy all files
shutil.copytree(path + "/raw-img", target_dir, dirs_exist_ok=True)

translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider"
}

for directory in os.listdir(target_dir):
    if directory in translate:
        new_name = translate[directory]
        if os.path.exists(target_dir / new_name):
            shutil.rmtree(target_dir / new_name)

        os.rename(target_dir / directory, target_dir / new_name)
        print(f"Renamed {directory} to {new_name}")

print("Copied to:", target_dir)
