# Task 2

Explores NLP and CV techniques with use of a publicly available [animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset.

## Solution Overview

This project is designed to verify claims about the presence or absence of specific animals in images, based on a given text description and an image. The solution combines Natural Language Processing (NLP) and Computer Vision (CV) techniques in a unified pipeline.

Supported animals: cat, dog, cow, horse, sheep, elephant, butterfly, chicken, spider, squirrel.

The pipeline consists of the following main steps:

1. **Named Entity Recognition (NER) on Text**
	- The text is processed using a custom NER model to extract animal entities mentioned in the sentence.
	- The model identifies both the animal and its position in the text.
2. **Negation Detection**
	- The context around the detected animal entity is analyzed to determine if the animal mention is negated (e.g., "not a cat").
	- A set of negation words ("not", "no", "never", etc.) is used to check for negation within a window of tokens before the animal.
3. **Image Classification**
	- The input image is classified using a ResNet-based model trained to recognize the same set of animals.
	- The model outputs the predicted animal class present in the image.
4. **Claim Verification**
	- The pipeline compares the animal detected in the text (and its negation status) with the animal predicted in the image.
	- If the text claims the animal is present (not negated) and the image matches, or if the text claims the animal is absent (negated) and the image does not match, the claim is considered verified (True). Otherwise, it is not verified (False).

## Key Components

- **NER Model**: Located in `models/ner/`, used via `ner/infer.py`.
- **Vision Model**: Located in `models/vision/`, used via `vision/infer.py`.
- **Pipeline**: Main logic in `pipeline/pipeline.py`.

## Example Usage

```python
from pipeline.pipeline import verify_text_image_claim

text = "There is a cow in the picture."
image_path = "test_images/cow/0001.jpeg"
result = verify_text_image_claim(text, image_path)
print(result)  # True or False
```

## Directory Structure

- `data/` — Contains training and validation data for NER and images for animals.
- `models/` — Stores trained NER and vision models.
- `ner/` — NER model code and inference logic.
- `vision/` — Vision model code, inference, and translation mapping.
- `pipeline/` — Main pipeline logic for claim verification.
- `test_images/` — Example images for each animal class.
- `utils/` — Utility scripts (e.g., data generation, negation handling).

## Getting Started

Follow these steps to set up and run the project:

1. **Install Python 3**: Ensure Python 3 is installed on your system.

1. **Clone the repository**
	```bash
    git clone https://github.com/OlyaFilatova/ds-practice.git
    cd ds-practice/task_2
	```

1. **Create and activate a virtual environment**
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```

1. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```

1. **Ensure model files are present**
	- The required NER and vision model files should be in the `models/ner/` and `models/vision/` directories, respectively.
	- If not present, download or train the models.
        - to train the ner model run
            
			```bash
			python utils/generate_ner_data.py
            python ner/train.py
			```
        - to train the vision model run 

			```bash
            python utils/download_vision_data.py
            python vision/train.py
			```
        - the pretrained vision model can be downloaded from https://drive.google.com/file/d/1IGw28vx8NhbAhXf0bghXybFt-MAjFfxE/view?usp=drive_link

1. **Running the pipeline**
	```python
	from pipeline.pipeline import verify_text_image_claim
	text = "There is a cow in the picture."
	image_path = "test_images/cow/0001.jpeg"
	result = verify_text_image_claim(text, image_path)
	print(result)  # True or False
	```

1. **(Optional) Explore notebooks and playground scripts**
	- See the `notebooks/` folder for examples and exploratory code.
