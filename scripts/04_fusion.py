"""Step 4: Fuse audio and text features"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fusion.fusion import fuse_features
from src.utils.config import Config


def main():
    """Fuse audio and text features"""
    print("=" * 60)
    print("Step 4: Feature Fusion")
    print("=" * 60)
    
    Config.create_directories()
    
    # Check if required files exist
    if not Config.AUDIO_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Audio features not found: {Config.AUDIO_FEATURES_CSV}\n"
            f"Run 02_extract_audio_features.py first."
        )
    
    # Fuse features
    df_fused = fuse_features()
    
    print("\n" + "=" * 60)
    print("Step 4 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
