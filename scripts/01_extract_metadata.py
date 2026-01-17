"""Step 1: Extract metadata and labels from DAIC-WOZ dataset"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.utils.config import Config


def main():
    """Extract metadata from DAIC-WOZ dataset"""
    print("=" * 60)
    print("Step 1: Extract Metadata")
    print("=" * 60)
    
    Config.create_directories()
    
    # Initialize loader
    loader = DataLoader()
    
    # Check if data needs extraction
    if not Config.SESSION_DATA_DIR.exists() or len(list(Config.SESSION_DATA_DIR.glob("*"))) == 0:
        print("\nData not extracted. Extracting ZIP files...")
        loader.extract_all_zips(force=False)
    else:
        print(f"\nData already exists in {Config.SESSION_DATA_DIR}")
        print("Skipping extraction. Use force=True in extract_all_zips() to re-extract.")
    
    # Load and save metadata
    print("\nLoading metadata...")
    loader.save_metadata()
    
    print("\n" + "=" * 60)
    print("Step 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
