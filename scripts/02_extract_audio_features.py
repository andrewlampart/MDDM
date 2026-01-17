"""Step 2: Extract audio features from DAIC-WOZ audio files"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.audio_extractor import AudioFeatureExtractor
from src.utils.config import Config


def main():
    """Extract audio features from all sessions"""
    print("=" * 60)
    print("Step 2: Extract Audio Features")
    print("=" * 60)
    
    Config.create_directories()
    
    # Load metadata
    if not Config.LABELS_CSV.exists():
        raise FileNotFoundError(
            f"Metadata not found: {Config.LABELS_CSV}\n"
            f"Run 01_extract_metadata.py first."
        )
    
    df = pd.read_csv(Config.LABELS_CSV)
    print(f"Loaded {len(df)} sessions from metadata")
    
    # Initialize extractor
    use_opensmile = Config.USE_OPENSMILE
    extractor = AudioFeatureExtractor(sr=Config.SAMPLE_RATE, use_opensmile=use_opensmile)
    
    if use_opensmile and extractor.use_opensmile:
        print("Using openSMILE for high-level features")
    else:
        print("openSMILE not available, using only low-level features")
    
    # Process all audio files
    audio_features = []
    failed_sessions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        session_id = row['session_id']
        audio_path = Path(row['audio_path'])
        
        if not audio_path.exists():
            print(f"\nWarning: Audio file not found for session {session_id}: {audio_path}")
            failed_sessions.append(session_id)
            continue
        
        try:
            # Extract features
            features = extractor.process_audio(audio_path)
            
            # Flatten for DataFrame
            flat_features = extractor.flatten_features(features)
            flat_features['session_id'] = session_id
            
            audio_features.append(flat_features)
        except Exception as e:
            print(f"\nError processing session {session_id}: {e}")
            failed_sessions.append(session_id)
            continue
    
    if failed_sessions:
        print(f"\nFailed to process {len(failed_sessions)} sessions: {failed_sessions}")
    
    # Create DataFrame and save
    if audio_features:
        df_audio = pd.DataFrame(audio_features)
        Config.AUDIO_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_audio.to_csv(Config.AUDIO_FEATURES_CSV, index=False)
        
        print(f"\nExtracted audio features for {len(df_audio)} sessions")
        print(f"Total audio features: {len(df_audio.columns) - 1}")
        print(f"Saved to: {Config.AUDIO_FEATURES_CSV}")
    else:
        print("\nError: No audio features extracted!")
    
    print("\n" + "=" * 60)
    print("Step 2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
