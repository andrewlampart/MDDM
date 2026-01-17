"""Extract metadata and transcriptions from DAIC-WOZ XML files"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional


def extract_labels_from_xml(xml_path: Path) -> Dict:
    """Ekstrakcja PHQ-8 score oraz binary depression label z XML
    
    Args:
        xml_path: Ścieżka do pliku XML z metadanymi sesji
        
    Returns:
        Dictionary z phq8_score, depression (0/1), qids_score
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    session = root.find('Session')
    if session is None:
        raise ValueError(f"No 'Session' element found in {xml_path}")
    
    phq8_elem = session.find('PHQ8_Score')
    if phq8_elem is None or phq8_elem.text is None:
        raise ValueError(f"No PHQ8_Score found in {xml_path}")
    
    phq8_score = int(phq8_elem.text)
    depression = int(phq8_score >= 10)  # PHQ-8 >= 10 = depresja
    
    qids_elem = session.find('QIDS_Score')
    qids_score = int(qids_elem.text) if qids_elem is not None and qids_elem.text else None
    
    return {
        'phq8_score': phq8_score,
        'depression': depression,
        'qids_score': qids_score
    }


def load_transcription(xml_path: Path) -> Optional[str]:
    """Ekstrakcja tekstu z pliku transkrypcji
    
    Args:
        xml_path: Ścieżka do pliku XML z transkrypcją
        
    Returns:
        Połączony tekst wypowiedzi uczestnika lub None jeśli brak
    """
    if not xml_path.exists():
        return None
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        dialogues = []
        for turn in root.findall('.//Turn'):
            speaker = turn.get('spk')  # 'Participant' lub 'Ellie'
            text_elem = turn.find('Text')
            text = text_elem.text if text_elem is not None else ''
            
            if speaker == 'Participant' and text:
                dialogues.append(text.strip())
        
        return ' '.join(dialogues) if dialogues else None
    except Exception as e:
        print(f"Error loading transcription from {xml_path}: {e}")
        return None
