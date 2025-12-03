import os
import eyed3

def edit_mp3_metadata_eyed3(file_path, lesson: str):
    """
    Edits the ID3 tags of an MP3 file using eyeD3.
    """
    print(file_path)
    try:
        audiofile = eyed3.load(file_path)
        if not audiofile:
            raise Exception("File was not loaded.")
        if audiofile.tag is None:
            audiofile.initTag()

        audiofile.tag.album = lesson

        audiofile.tag.save()
        print(f"Metadata updated for {file_path}")
    except Exception as e:
        print(f"Error updating metadata for {file_path}: {e}")

root_dir = f"./audio"

dirs = [directory for directory in os.listdir(root_dir) if os.path.isdir(f"{root_dir}/{directory}")]

for directory in dirs:
    directory_path = f"{root_dir}/{directory}"
    files = [directory for directory in os.listdir(directory_path) if not os.path.isdir(f"{directory_path}/{directory}") and directory.endswith(".mp3")]
    [edit_mp3_metadata_eyed3(f"{directory_path}/{file}", directory) for file in files]
