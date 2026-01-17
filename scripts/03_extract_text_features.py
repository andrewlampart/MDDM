"""Step 3: Extract text features from DAIC-WOZ transcriptions"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.text_extractor import TextFeatureExtractor
from src.utils.config import Config


def main():
    """Extract text features from all transcriptions"""
    print("=" * 60)
    print("Step 3: Extract Text Features")
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
    
    # Filter sessions with text
    df_with_text = df[df['text'].notna() & (df['text'].str.strip() != '')]
    print(f"Sessions with transcriptions: {len(df_with_text)}")
    
    if len(df_with_text) == 0:
        print("Warning: No transcriptions found. Text features will be empty.")
        # Create empty DataFrame with correct structure
        text_extractor = TextFeatureExtractor()
        empty_features = text_extractor.flatten_features(None, session_id=0)
        empty_features.pop('session_id')  # Remove session_id from structure
        df_text = pd.DataFrame([empty_features])
        df_text['session_id'] = df['session_id'].values[:1]  # Just for structure
    else:
        # Initialize extractor
        print("Loading transformer model...")
        text_extractor = TextFeatureExtractor(model_name=Config.TEXT_MODEL_NAME)
        print("Model loaded successfully")
        
        # Process all transcriptions
        text_features = []
        failed_sessions = []
        
        for idx, row in tqdm(df_with_text.iterrows(), total=len(df_with_text), desc="Processing text"):
            session_id = row['session_id']
            text = row['text']
            
            try:
                # Extract features
                features = text_extractor.process_text(text)
                
                # Flatten for DataFrame
                flat_features = text_extractor.flatten_features(features, session_id=session_id)
                
                text_features.append(flat_features)
            except Exception as e:
                print(f"\nError processing session {session_id}: {e}")
                failed_sessions.append(session_id)
                continue
        
        if failed_sessions:
            print(f"\nFailed to process {len(failed_sessions)} sessions: {failed_sessions}")
        
        # Create DataFrame
        if text_features:
            df_text = pd.DataFrame(text_features)
        else:
            print("Warning: No text features extracted!")
            # Create empty structure
            empty_features = text_extractor.flatten_features(None, session_id=0)
            df_text = pd.DataFrame([empty_features])
    
    # Add sessions without text (filled with zeros)
    sessions_without_text = df[~df['session_id'].isin(df_text['session_id'].values)]
    if len(sessions_without_text) > 0:
        print(f"\nAdding {len(sessions_without_text)} sessions without text (filled with zeros)")
        text_extractor = TextFeatureExtractor()
        for _, row in sessions_without_text.iterrows():
            empty_features = text_extractor.flatten_features(None, session_id=row['session_id'])
            df_text = pd.concat([df_text, pd.DataFrame([empty_features])], ignore_index=True)
    
    # Save
    Config.TEXT_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_text.to_csv(Config.TEXT_FEATURES_CSV, index=False)
    
    print(f"\nExtracted text features for {len(df_text)} sessions")
    print(f"Total text features: {len(df_text.columns) - 1}")
    print(f"Saved to: {Config.TEXT_FEATURES_CSV}")
    
    print("\n" + "=" * 60)
    print("Step 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
