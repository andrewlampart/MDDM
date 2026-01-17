"""Interaktywny skrypt do uruchamiania poszczególnych kroków pipeline"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config


def check_step_completed(step_num: int) -> bool:
    """Sprawdza czy krok jest już wykonany"""
    if step_num == 1:
        return Config.LABELS_CSV.exists()
    elif step_num == 2:
        return Config.AUDIO_FEATURES_CSV.exists()
    elif step_num == 3:
        return Config.TEXT_FEATURES_CSV.exists()
    elif step_num == 4:
        return Config.FUSED_FEATURES_H5.exists() or Config.FUSED_FEATURES_H5.with_suffix('.csv').exists()
    elif step_num == 5:
        return Config.MODEL_PKL.exists() and Config.SCALER_PKL.exists()
    return False


def get_step_name(step_num: int) -> str:
    """Zwraca nazwę kroku"""
    names = {
        1: "Ekstrakcja metadanych",
        2: "Ekstrakcja cech audio",
        3: "Ekstrakcja cech tekstowych",
        4: "Fuzja cech",
        5: "Trening modelu"
    }
    return names.get(step_num, "Nieznany krok")


def get_step_script(step_num: int) -> Path:
    """Zwraca ścieżkę do skryptu kroku"""
    scripts_dir = Path(__file__).parent
    scripts = {
        1: scripts_dir / "01_extract_metadata.py",
        2: scripts_dir / "02_extract_audio_features.py",
        3: scripts_dir / "03_extract_text_features.py",
        4: scripts_dir / "04_fusion.py",
        5: scripts_dir / "05_train_model.py"
    }
    return scripts.get(step_num)


def print_menu():
    """Wyświetla menu z checkboxami"""
    print("\n" + "=" * 60)
    print("MDDM Pipeline - Menu")
    print("=" * 60)
    print()
    
    for i in range(1, 6):
        status = "✓" if check_step_completed(i) else "✗"
        print(f"  [{status}] {i}. {get_step_name(i)}")
    
    print()
    print("  6. Uruchom pełny pipeline")
    print("  0. Wyjście")
    print("=" * 60)


def run_step(step_num: int):
    """Uruchamia wybrany krok"""
    script_path = get_step_script(step_num)
    
    if not script_path.exists():
        print(f"\n❌ Błąd: Nie znaleziono skryptu {script_path}")
        return
    
    print(f"\n▶ Uruchamianie: {get_step_name(step_num)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            check=True
        )
        print("\n✅ Krok zakończony pomyślnie!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Błąd podczas wykonywania kroku: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠ Przerwano przez użytkownika")


def run_full_pipeline():
    """Uruchamia pełny pipeline"""
    script_path = Path(__file__).parent / "run_full_pipeline.py"
    
    if not script_path.exists():
        print(f"\n❌ Błąd: Nie znaleziono skryptu {script_path}")
        return
    
    print("\n▶ Uruchamianie pełnego pipeline...")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root,
            check=True
        )
        print("\n✅ Pipeline zakończony pomyślnie!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Błąd podczas wykonywania pipeline: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠ Przerwano przez użytkownika")


def main():
    """Główna pętla menu"""
    while True:
        print_menu()
        
        try:
            choice = input("\nWybierz opcję (0-6): ").strip()
            
            if choice == "0":
                print("\nDo widzenia!")
                break
            elif choice in ["1", "2", "3", "4", "5"]:
                step_num = int(choice)
                run_step(step_num)
            elif choice == "6":
                confirm = input("\n⚠ Uruchomić pełny pipeline? (t/n): ").strip().lower()
                if confirm == "t":
                    run_full_pipeline()
                else:
                    print("Anulowano.")
            else:
                print("\n❌ Nieprawidłowa opcja. Wybierz 0-6.")
        
        except KeyboardInterrupt:
            print("\n\nDo widzenia!")
            break
        except Exception as e:
            print(f"\n❌ Błąd: {e}")


if __name__ == "__main__":
    main()
