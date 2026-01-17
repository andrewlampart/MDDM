"""Pipeline menu for MDDM PhD Project

Multimodal Depression Detection Model - Full Pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try new config first, fall back to old
try:
    from config import CONFIG as Config
    Config.create_directories = lambda: Config.__post_init__(Config)
except ImportError:
    from src.utils.config import Config


def clear_screen():
    """Czyści ekran terminala"""
    os.system('cls' if os.name == 'nt' else 'clear')


def check_status(step_num: int) -> bool:
    """Sprawdza czy krok jest wykonany"""
    results_dir = project_root / "results"
    models_dir = project_root / "data" / "models"
    processed_dir = project_root / "data" / "processed"
    
    checks = {
        1: (processed_dir / "daic_labels.csv").exists(),
        2: (processed_dir / "audio_features.csv").exists(),
        3: (processed_dir / "text_features.csv").exists(),
        4: (processed_dir / "daic_fused_features.csv").exists() or (processed_dir / "daic_fused_features.h5").exists(),
        5: (models_dir / "baseline_model.pkl").exists(),
        # Sprint 1
        10: (models_dir / "xgboost_baseline.pkl").exists(),
        # Sprint 2
        20: (processed_dir / "instances.csv").exists(),
        21: (models_dir / "mil_text_best.pt").exists(),
        # Sprint 3
        30: (models_dir / "audio_cnn_best.pt").exists(),
        31: (models_dir / "multimodal_best.pt").exists(),
        32: (results_dir / "model_comparison.csv").exists(),
    }
    return checks.get(step_num, False)


def show_menu():
    """Wyświetla menu z checkboxami"""
    clear_screen()
    
    header = """
╔══════════════════════════════════════════════════════════════════════╗
║     MDDM Pipeline - Multimodal Depression Detection PhD Framework    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(header)
    
    # Original pipeline
    print("  📁 Data Preprocessing:")
    steps = [
        (1, "Ekstrakcja metadanych"),
        (2, "Ekstrakcja cech audio"),
        (3, "Ekstrakcja cech tekstowych"),
        (4, "Fuzja cech (early)"),
        (5, "Trening baseline (RF)"),
    ]
    for num, name in steps:
        status = "✓" if check_status(num) else " "
        print(f"      [{status}] {num}. {name}")
    
    # Sprint 1
    print("\n  🎯 Sprint 1: XGBoost Baseline:")
    print(f"      [{'✓' if check_status(10) else ' '}] 10. XGBoost baseline + ewaluacja")
    
    # Sprint 2
    print("\n  📝 Sprint 2: MIL Text Model:")
    print(f"      [{'✓' if check_status(20) else ' '}] 20. Segmentacja wywiadów (MIL)")
    print(f"      [{'✓' if check_status(21) else ' '}] 21. Trening MIL Text (RoBERTa)")
    
    # Sprint 3
    print("\n  🔊 Sprint 3: Multimodal Fusion:")
    print(f"      [{'✓' if check_status(30) else ' '}] 30. Trening Audio CNN + LSTM")
    print(f"      [{'✓' if check_status(31) else ' '}] 31. Trening Late Fusion")
    print(f"      [{'✓' if check_status(32) else ' '}] 32. Porównanie wszystkich modeli")
    
    print("\n" + "─" * 70)
    print("  Batch operations:")
    print("      [A] Pełny preprocessing (1-5)")
    print("      [B] Pełny PhD pipeline (10, 20-21, 30-32)")
    print("      [Q] Wyjście")
    print("─" * 70)
    sys.stdout.flush()


def run_script(script_name: str, script_dir: str = "scripts") -> bool:
    """Uruchamia skrypt"""
    script_path = project_root / script_dir / script_name
    
    if not script_path.exists():
        print(f"\n❌ Nie znaleziono: {script_path}")
        input("\nNaciśnij Enter aby kontynuować...")
        return False
    
    print(f"\n{'='*70}")
    print(f"▶ Uruchamianie: {script_name}")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_root
        )
        
        if result.returncode == 0:
            print("\n✅ Gotowe!")
        else:
            print(f"\n⚠ Zakończono z kodem: {result.returncode}")
        
        input("\nNaciśnij Enter aby kontynuować...")
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Przerwano przez użytkownika")
        input("\nNaciśnij Enter aby kontynuować...")
        return False
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        input("\nNaciśnij Enter aby kontynuować...")
        return False


def main():
    """Główna pętla"""
    # Script mappings
    script_map = {
        "1": ("scripts", "01_extract_metadata.py"),
        "2": ("scripts", "02_extract_audio_features.py"),
        "3": ("scripts", "03_extract_text_features.py"),
        "4": ("scripts", "04_fusion.py"),
        "5": ("scripts", "05_train_model.py"),
        # Sprint 1
        "10": ("experiments", "run_baselines.py"),
        # Sprint 2
        "20": ("data", "interview_segmentation.py"),
        "21": ("experiments", "train_mil_text.py"),
        # Sprint 3
        "30": ("experiments", "train_multimodal.py"),
        "31": ("experiments", "train_multimodal.py"),
        "32": ("experiments", "compare_all_models.py"),
    }
    
    while True:
        show_menu()
        
        try:
            choice = input("\nWybierz opcję: ").strip().upper()
            
            if choice == "Q" or choice == "0":
                clear_screen()
                print("\nDo widzenia!\n")
                break
                
            elif choice == "A":
                confirm = input("\n⚠ Uruchomić preprocessing (1-5)? (t/n): ").strip().lower()
                if confirm == "t":
                    for num in ["1", "2", "3", "4", "5"]:
                        script_dir, script_name = script_map[num]
                        if not run_script(script_name, script_dir):
                            print("\n❌ Pipeline przerwany")
                            input("\nNaciśnij Enter aby kontynuować...")
                            break
                    else:
                        print("\n✅ Preprocessing zakończony!")
                        input("\nNaciśnij Enter aby kontynuować...")
                        
            elif choice == "B":
                confirm = input("\n⚠ Uruchomić PhD pipeline (10, 20-21, 30-32)? (t/n): ").strip().lower()
                if confirm == "t":
                    for num in ["10", "20", "21", "30", "32"]:
                        script_dir, script_name = script_map[num]
                        if not run_script(script_name, script_dir):
                            print("\n❌ Pipeline przerwany")
                            input("\nNaciśnij Enter aby kontynuować...")
                            break
                    else:
                        print("\n✅ PhD Pipeline zakończony!")
                        input("\nNaciśnij Enter aby kontynuować...")
                        
            elif choice in script_map:
                script_dir, script_name = script_map[choice]
                run_script(script_name, script_dir)
            else:
                print("\n❌ Nieprawidłowa opcja")
                input("\nNaciśnij Enter aby kontynuować...")
        
        except KeyboardInterrupt:
            clear_screen()
            print("\n\nDo widzenia!\n")
            break
        except Exception as e:
            print(f"\n❌ Błąd: {e}")
            input("\nNaciśnij Enter aby kontynuować...")


if __name__ == "__main__":
    main()
