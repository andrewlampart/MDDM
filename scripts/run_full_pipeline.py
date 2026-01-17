"""Run the complete pipeline from data extraction to model training"""

import sys
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_script(script_path: Path):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print(f"{'='*60}\n")
    
    spec = importlib.util.spec_from_file_location("script", script_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"Warning: {script_path.name} has no main() function")
    except Exception as e:
        print(f"\nError running {script_path.name}: {e}")
        raise


def main():
    """Run complete pipeline"""
    print("=" * 60)
    print("DAIC-WOZ Depression Detection - Full Pipeline")
    print("=" * 60)
    
    scripts_dir = Path(__file__).parent
    steps = [
        scripts_dir / "01_extract_metadata.py",
        scripts_dir / "02_extract_audio_features.py",
        scripts_dir / "03_extract_text_features.py",
        scripts_dir / "04_fusion.py",
        scripts_dir / "05_train_model.py",
    ]
    
    for step in steps:
        if not step.exists():
            print(f"Error: Script not found: {step}")
            continue
        
        try:
            run_script(step)
        except KeyboardInterrupt:
            print("\n\nPipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\nPipeline failed at {step.name}")
            print(f"Error: {e}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
