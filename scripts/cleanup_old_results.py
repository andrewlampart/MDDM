#!/usr/bin/env python
"""
Cleanup Script - Czyści stare wyniki przed nową analizą

Usuwa zawartość katalogów:
- data/processed (przetworzone features)
- data/models (zapisane modele)
- results (wyniki eksperymentów)
- logs (logi treningów)

Zachowuje pliki .gitkeep dla struktury git.
"""
import shutil
from pathlib import Path
import argparse


DIRS_TO_CLEAN = [
    "data/processed",
    "data/models",
    "results",
    "logs"
]

FILES_TO_KEEP = [".gitkeep", "__init__.py"]


def cleanup(root_dir: Path, dry_run: bool = False) -> dict:
    """
    Czyści stare wyniki z określonych katalogów.
    
    Args:
        root_dir: Ścieżka do katalogu głównego projektu
        dry_run: Jeśli True, tylko wyświetla co zostanie usunięte
        
    Returns:
        dict: Statystyki czyszczenia
    """
    stats = {
        "files_deleted": 0,
        "dirs_deleted": 0,
        "bytes_freed": 0
    }
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Rozpoczynam cleanup...")
    print(f"Root: {root_dir}\n")
    
    for dir_name in DIRS_TO_CLEAN:
        dir_path = root_dir / dir_name
        
        if not dir_path.exists():
            print(f"  [SKIP] Pomijam (nie istnieje): {dir_name}")
            continue
            
        print(f"  [DIR] Przetwarzam: {dir_name}")
        items_to_delete = []
        
        # Zbierz wszystkie elementy do usunięcia
        for item in dir_path.rglob("*"):
            # Pomijaj pliki do zachowania
            if item.name in FILES_TO_KEEP:
                continue
            # Pomijaj katalogi rodziców plików do zachowania
            if any(keep_file in str(item) for keep_file in FILES_TO_KEEP):
                continue
            items_to_delete.append(item)
        
        # Najpierw pliki, potem katalogi (od najgłębszych)
        files = [f for f in items_to_delete if f.is_file()]
        dirs = sorted([d for d in items_to_delete if d.is_dir()], 
                     key=lambda x: len(x.parts), reverse=True)
        
        # Usuń pliki
        for file_path in files:
            try:
                size = file_path.stat().st_size
                if dry_run:
                    print(f"    [DRY] {file_path.relative_to(root_dir)} ({size:,} bytes)")
                else:
                    file_path.unlink()
                    print(f"    [OK] Usunieto: {file_path.relative_to(root_dir)}")
                stats["files_deleted"] += 1
                stats["bytes_freed"] += size
            except Exception as e:
                print(f"    [ERR] Blad przy {file_path}: {e}")
        
        # Usuń katalogi (tylko puste po usunięciu plików)
        for dir_to_del in dirs:
            try:
                if dry_run:
                    print(f"    [DRY DIR] {dir_to_del.relative_to(root_dir)}/")
                else:
                    # Spróbuj usunąć - zadziała tylko jeśli pusty
                    try:
                        dir_to_del.rmdir()
                        print(f"    [OK] Usunieto katalog: {dir_to_del.relative_to(root_dir)}")
                        stats["dirs_deleted"] += 1
                    except OSError:
                        # Katalog nie jest pusty - użyj shutil
                        shutil.rmtree(dir_to_del)
                        print(f"    [OK] Usunieto katalog (z zawartoscia): {dir_to_del.relative_to(root_dir)}")
                        stats["dirs_deleted"] += 1
            except Exception as e:
                print(f"    [ERR] Blad przy katalogu {dir_to_del}: {e}")
        
        print()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Czyści stare wyniki przed nową analizą"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Tylko pokaż co zostanie usunięte, bez faktycznego usuwania"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Nie pytaj o potwierdzenie"
    )
    args = parser.parse_args()
    
    # Znajdź katalog główny projektu
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    print("=" * 60)
    print("MDDM CLEANUP SCRIPT")
    print("=" * 60)
    print(f"\nKatalogi do wyczyszczenia:")
    for d in DIRS_TO_CLEAN:
        path = root_dir / d
        exists = "[OK]" if path.exists() else "[--]"
        print(f"  {exists} {d}")
    print(f"\nPliki zachowane: {', '.join(FILES_TO_KEEP)}")
    print()
    
    # Potwierdzenie (chyba że --yes lub --dry-run)
    if not args.yes and not args.dry_run:
        confirm = input("Czy kontynuować? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes", "tak", "t"):
            print("Anulowano.")
            return
    
    stats = cleanup(root_dir, dry_run=args.dry_run)
    
    print("=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(f"Usunięto plików:      {stats['files_deleted']}")
    print(f"Usunięto katalogów:   {stats['dirs_deleted']}")
    print(f"Zwolniono przestrzeni: {stats['bytes_freed']:,} bytes ({stats['bytes_freed'] / 1024 / 1024:.2f} MB)")
    print()
    
    if args.dry_run:
        print("[INFO] To byl dry run. Uruchom bez --dry-run aby faktycznie usunac.")
    else:
        print("[OK] Cleanup zakonczony pomyslnie!")


if __name__ == "__main__":
    main()
