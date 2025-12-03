# Simplest scraping example

Loads page, reads list of audio files for source elements, stores audio files.

## Running the solution

### Add links.txt file

In folder audio add links.txt file that contains list of separated by new line urls of pubilcly available pages that have audio players in them.

Note: Manually check robots.txt and website policies for scraping limitations.

### Setup virtual environment

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download audio files

```python download.py```

### Update audio file metadata

```python add_audio_meta.py```
