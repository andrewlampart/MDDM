"""
Interview Segmentation for Multi-Instance Learning

Segments DAIC-WOZ interviews into instances (patient responses).
Each response becomes one instance in the MIL framework.

The bag-level label (depression) is assigned to all instances from the same interview.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG


class InterviewSegmenter:
    """
    Segments interviews into instances for Multi-Instance Learning.
    
    Strategies:
    1. QA-pairs: Each patient response (from transcript CSV) = 1 instance
    2. Sentences: Split full text into sentences
    3. Chunks: Fixed-length text chunks
    """
    
    def __init__(self, min_words: int = 3, max_words: int = 512):
        """
        Args:
            min_words: Minimum words per instance
            max_words: Maximum words per instance (for transformer limit)
        """
        self.min_words = min_words
        self.max_words = max_words
    
    def segment_from_transcript_csv(self, csv_path: Path, 
                                     speaker_filter: str = 'Participant') -> List[Dict]:
        """
        Segment interview from transcript CSV file.
        
        DAIC-WOZ transcript format:
        start_time, stop_time, speaker, value
        
        Args:
            csv_path: Path to transcript CSV
            speaker_filter: Speaker to extract (usually 'Participant')
            
        Returns:
            List of instance dictionaries with text and timing info
        """
        instances = []
        
        try:
            # Try different separators
            df = None
            for sep in ['\t', ',']:
                try:
                    df = pd.read_csv(csv_path, sep=sep)
                    if len(df.columns) >= 4:
                        break
                except:
                    continue
            
            if df is None or len(df.columns) < 4:
                return []
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Find relevant columns
            speaker_col = None
            text_col = None
            start_col = None
            stop_col = None
            
            for col in df.columns:
                if 'speaker' in col:
                    speaker_col = col
                elif 'value' in col or 'text' in col:
                    text_col = col
                elif 'start' in col:
                    start_col = col
                elif 'stop' in col or 'end' in col:
                    stop_col = col
            
            if speaker_col is None or text_col is None:
                return []
            
            # Filter by speaker
            df_filtered = df[df[speaker_col].astype(str).str.contains(
                speaker_filter, case=False, na=False
            )]
            
            sequence_num = 0
            for idx, row in df_filtered.iterrows():
                text = str(row[text_col]).strip()
                
                # Skip empty or very short responses
                if not text or text.lower() in ['nan', 'none', '']:
                    continue
                
                word_count = len(text.split())
                if word_count < self.min_words:
                    continue
                
                # Truncate if too long
                if word_count > self.max_words:
                    words = text.split()[:self.max_words]
                    text = ' '.join(words)
                
                instance = {
                    'sequence_num': sequence_num,
                    'text': text,
                    'word_count': len(text.split()),
                }
                
                # Add timing info if available
                if start_col and start_col in row:
                    try:
                        instance['start_time'] = float(row[start_col])
                    except:
                        pass
                
                if stop_col and stop_col in row:
                    try:
                        instance['end_time'] = float(row[stop_col])
                    except:
                        pass
                
                instances.append(instance)
                sequence_num += 1
        
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
        
        return instances
    
    def segment_by_sentences(self, text: str) -> List[Dict]:
        """
        Segment text into sentence-level instances.
        
        Args:
            text: Full interview text
            
        Returns:
            List of instance dictionaries
        """
        if not text or text.strip() == '':
            return []
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        instances = []
        buffer = ""
        sequence_num = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Combine short sentences
            potential = (buffer + " " + sentence).strip() if buffer else sentence
            word_count = len(potential.split())
            
            if word_count > self.max_words:
                # Save buffer if not empty
                if buffer and len(buffer.split()) >= self.min_words:
                    instances.append({
                        'sequence_num': sequence_num,
                        'text': buffer.strip(),
                        'word_count': len(buffer.split())
                    })
                    sequence_num += 1
                buffer = sentence
            elif word_count >= self.min_words:
                buffer = potential
            else:
                buffer = potential
        
        # Don't forget the last buffer
        if buffer and len(buffer.split()) >= self.min_words:
            instances.append({
                'sequence_num': sequence_num,
                'text': buffer.strip(),
                'word_count': len(buffer.split())
            })
        
        return instances
    
    def segment_by_chunks(self, text: str, chunk_size: int = 100, 
                         overlap: int = 20) -> List[Dict]:
        """
        Segment text into fixed-size word chunks with overlap.
        
        Args:
            text: Full interview text
            chunk_size: Words per chunk
            overlap: Overlapping words between chunks
            
        Returns:
            List of instance dictionaries
        """
        if not text or text.strip() == '':
            return []
        
        words = text.split()
        instances = []
        
        start = 0
        sequence_num = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            
            if len(chunk_words) >= self.min_words:
                instances.append({
                    'sequence_num': sequence_num,
                    'text': ' '.join(chunk_words),
                    'word_count': len(chunk_words)
                })
                sequence_num += 1
            
            start += chunk_size - overlap
            if start >= len(words):
                break
        
        return instances


def create_mil_instances_dataset(
    output_path: Path = None,
    segmentation_method: str = 'qa_pairs'
) -> pd.DataFrame:
    """
    Create MIL instances dataset from DAIC-WOZ transcripts.
    
    Args:
        output_path: Path to save instances CSV
        segmentation_method: 'qa_pairs', 'sentences', or 'chunks'
        
    Returns:
        DataFrame with columns: session_id, sequence_num, text, word_count, depression
    """
    output_path = output_path or CONFIG.INSTANCES_CSV
    
    print("=" * 60)
    print("Creating MIL Instances Dataset")
    print("=" * 60)
    print(f"Segmentation method: {segmentation_method}")
    
    # Load labels
    labels_df = pd.read_csv(CONFIG.LABELS_CSV)
    labels_dict = dict(zip(labels_df['session_id'], labels_df['depression']))
    
    # Also get text from labels if available
    if 'text' in labels_df.columns:
        text_dict = dict(zip(labels_df['session_id'], labels_df['text']))
    else:
        text_dict = {}
    
    segmenter = InterviewSegmenter()
    all_instances = []
    
    sessions_processed = 0
    sessions_skipped = 0
    
    for session_id, depression_label in tqdm(labels_dict.items(), desc="Processing sessions"):
        instances = []
        
        if segmentation_method == 'qa_pairs':
            # Try to find transcript CSV
            transcript_path = CONFIG.RAW_DATA_DIR / f"{session_id}_TRANSCRIPT.csv"
            
            if transcript_path.exists():
                instances = segmenter.segment_from_transcript_csv(transcript_path)
            else:
                # Fall back to text from labels
                if session_id in text_dict and pd.notna(text_dict[session_id]):
                    text = str(text_dict[session_id])
                    instances = segmenter.segment_by_sentences(text)
        
        elif segmentation_method == 'sentences':
            if session_id in text_dict and pd.notna(text_dict[session_id]):
                text = str(text_dict[session_id])
                instances = segmenter.segment_by_sentences(text)
        
        elif segmentation_method == 'chunks':
            if session_id in text_dict and pd.notna(text_dict[session_id]):
                text = str(text_dict[session_id])
                instances = segmenter.segment_by_chunks(text)
        
        if not instances:
            sessions_skipped += 1
            continue
        
        # Add session info to each instance
        for inst in instances:
            inst['session_id'] = session_id
            inst['depression'] = depression_label
        
        all_instances.extend(instances)
        sessions_processed += 1
    
    # Create DataFrame
    df_instances = pd.DataFrame(all_instances)
    
    # Reorder columns
    cols = ['session_id', 'sequence_num', 'text', 'word_count', 'depression']
    cols = [c for c in cols if c in df_instances.columns]
    df_instances = df_instances[cols]
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_instances.to_csv(output_path, index=False)
    
    # Statistics
    print(f"\n{'─'*60}")
    print("Dataset Statistics:")
    print(f"{'─'*60}")
    print(f"  Sessions processed: {sessions_processed}")
    print(f"  Sessions skipped (no text): {sessions_skipped}")
    print(f"  Total instances: {len(df_instances)}")
    print(f"  Instances per session: {len(df_instances) / sessions_processed:.1f} avg")
    print(f"  Unique sessions: {df_instances['session_id'].nunique()}")
    
    # Class distribution
    session_labels = df_instances.groupby('session_id')['depression'].first()
    print(f"\n  Class distribution (session-level):")
    print(f"    Depressed: {session_labels.sum()} ({session_labels.mean():.1%})")
    print(f"    Non-depressed: {(~session_labels.astype(bool)).sum()} ({1-session_labels.mean():.1%})")
    
    print(f"\n  Saved to: {output_path}")
    print("=" * 60)
    
    return df_instances


def get_bag_statistics(instances_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics for each bag (interview).
    
    Args:
        instances_df: DataFrame from create_mil_instances_dataset
        
    Returns:
        DataFrame with bag-level statistics
    """
    stats = instances_df.groupby('session_id').agg({
        'text': 'count',
        'word_count': ['sum', 'mean', 'std'],
        'depression': 'first'
    })
    
    stats.columns = ['n_instances', 'total_words', 'mean_words', 'std_words', 'depression']
    stats = stats.reset_index()
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create MIL instances dataset')
    parser.add_argument('--method', type=str, default='qa_pairs',
                       choices=['qa_pairs', 'sentences', 'chunks'],
                       help='Segmentation method')
    args = parser.parse_args()
    
    df = create_mil_instances_dataset(segmentation_method=args.method)
    
    # Print sample
    print("\nSample instances:")
    print(df.head(10).to_string())
