from pathlib import Path
import gdown
import shutil

# 1. Update this to match where your API actually looks for files
LOCAL_DATA = Path("policies") 
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1ijOsUeSUfeSwHaG4zQpMKi0KWpAb7CSs"


def is_download_complete(path: Path) -> bool:
    """Strict sanity check: folder exists, is a directory, and contains actual files."""
    if not path.exists() or not path.is_dir():
        return False
    # Ensure it contains actual files (ignoring hidden files like .gitkeep or empty folders)
    files = [f for f in path.rglob("*") if f.is_file() and not f.name.startswith('.')]
    return len(files) > 0


def download_assets(force: bool = False):
    """
    Downloads assets from Google Drive folder.
    
    Args:
        force (bool): If True, re-downloads even if data exists.
    """

    # Fix: Now correctly evaluates if files are missing inside ./policies
    if not force and is_download_complete(LOCAL_DATA):
        print("Assets already downloaded and verified.")
        return

    # Clean broken downloads if forcing or corrupted
    if force and LOCAL_DATA.exists():
        shutil.rmtree(LOCAL_DATA)

    LOCAL_DATA.mkdir(parents=True, exist_ok=True)

    print(f"Downloading policy assets into {LOCAL_DATA}...")

    try:
        gdown.download_folder(
            url=DRIVE_FOLDER,
            output=str(LOCAL_DATA),
            quiet=False,
            use_cookies=True,  # more reliable for Drive folders
        )
        print("Download complete.")

    except Exception as e:
        print(f"Download failed: {e}")
        raise