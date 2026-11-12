"""
Download CLEVR-Change Dataset from Google Drive

This script downloads the CLEVR-Change dataset from the official
Robust Change Captioning (ICCV 2019) repository.

Dataset Details:
- Source: https://github.com/Seth-Park/RobustChangeCaptioning
- Size: ~2.5 GB (compressed)
- Contains: 79,606 image pairs with change captions
- 5 change types: add, drop, move, color, material
- Split: train/val/test
"""

import os
import sys
import requests
from tqdm import tqdm

def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        destination: Local file path to save
    """
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    # Check if we need to handle the virus scan warning
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            break
    
    # Download with progress bar
    print(f"Downloading to {destination}...")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Progress") as pbar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Download complete: {destination}")

def extract_tar_gz(tar_path, extract_dir):
    """Extract tar.gz file."""
    import tarfile
    
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Get total number of members for progress bar
        members = tar.getmembers()
        
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, path=extract_dir)
                pbar.update(1)
    
    print(f"Extraction complete: {extract_dir}")

def main():
    # Configuration
    FILE_ID = '1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe'  # CLEVR-Change dataset
    FILENAME = 'clevr_change.tar.gz'
    DATA_DIR = './data'
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    destination = os.path.join(DATA_DIR, FILENAME)
    
    # Check if already downloaded
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            if os.path.exists(os.path.join(DATA_DIR, 'images')):
                print("Dataset appears to be already extracted.")
                return
        else:
            os.remove(destination)
    
    # Download
    try:
        download_file_from_google_drive(FILE_ID, destination)
    except Exception as e:
        print(f"Error during download: {e}")
        sys.exit(1)
    
    # Extract
    try:
        extract_tar_gz(destination, '.')
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)
    
    # Verify extraction
    if os.path.exists(os.path.join(DATA_DIR, 'images')):
        print("\n✓ Dataset successfully downloaded and extracted!")
        print(f"\nDataset structure:")
        print(f"  {DATA_DIR}/")
        print(f"    ├── images/           (default images)")
        print(f"    ├── sc_images/        (semantically changed images)")
        print(f"    ├── nsc_images/       (non-semantically changed images)")
        print(f"    ├── change_captions.json")
        print(f"    ├── no_change_captions.json")
        print(f"    └── splits.json")
        
        # Count files
        num_images = len([f for f in os.listdir(os.path.join(DATA_DIR, 'images')) if f.endswith('.png')])
        print(f"\nTotal images: {num_images}")
        
        print("\n✓ Ready to proceed with Phase 5!")
        print("\nNext steps:")
        print("  1. Preprocess dataset (extract features or use raw images)")
        print("  2. Train baseline M3Diff model")
        print("  3. Implement and test improvements")
        
    else:
        print("\n✗ Error: Dataset extraction may have failed.")
        print("Please check the data directory manually.")

if __name__ == '__main__':
    print("=" * 60)
    print("CLEVR-Change Dataset Downloader")
    print("Phase 5 - Experimental Validation")
    print("=" * 60)
    print()
    
    main()
