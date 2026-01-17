"""
Experiment Tracker for MDDM Pipeline

Prosty system trackingu eksperymentów bez zewnętrznych zależności.
Zapisuje parametry, metryki i artefakty każdego runu.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG


@dataclass
class RunInfo:
    """Informacje o pojedynczym runie eksperymentu"""
    run_id: str
    model_name: str
    timestamp: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]
    duration_seconds: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ExperimentTracker:
    """
    Prosty tracker eksperymentów zapisujący wyniki do JSON.
    
    Użycie:
        tracker = ExperimentTracker("my_experiment")
        tracker.start_run("XGBoost")
        tracker.log_params({'n_estimators': 100, 'max_depth': 6})
        tracker.log_metrics({'f1': 0.65, 'auroc': 0.72})
        tracker.log_artifact('model.pkl')
        tracker.end_run()
    """
    
    def __init__(self, experiment_name: str, base_dir: Path = None):
        """
        Args:
            experiment_name: Nazwa eksperymentu
            base_dir: Katalog bazowy dla logów (default: CONFIG.LOGS_DIR)
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir or CONFIG.LOGS_DIR)
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs_file = self.experiment_dir / "runs.json"
        self.runs: List[RunInfo] = []
        
        # Current run state
        self._current_run: Optional[RunInfo] = None
        self._current_run_dir: Optional[Path] = None
        self._run_start_time: Optional[datetime] = None
        
        # Load existing runs
        self._load_runs()
        
        print(f"ExperimentTracker initialized: {self.experiment_dir}")
        print(f"  Existing runs: {len(self.runs)}")
    
    def _load_runs(self):
        """Wczytaj historię runów z pliku JSON"""
        if self.runs_file.exists():
            try:
                with open(self.runs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.runs = [RunInfo(**r) for r in data.get('runs', [])]
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load runs history: {e}")
                self.runs = []
    
    def _save_runs(self):
        """Zapisz historię runów do pliku JSON"""
        data = {
            'experiment_name': self.experiment_name,
            'last_updated': datetime.now().isoformat(),
            'total_runs': len(self.runs),
            'runs': [r.to_dict() for r in self.runs]
        }
        
        with open(self.runs_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _generate_run_id(self, model_name: str) -> str:
        """Generuj unikalny ID runu"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        return f"{safe_name}_{timestamp}"
    
    def start_run(self, model_name: str, notes: str = "") -> str:
        """
        Rozpocznij nowy run eksperymentu.
        
        Args:
            model_name: Nazwa modelu
            notes: Opcjonalne notatki
            
        Returns:
            run_id
        """
        if self._current_run is not None:
            print(f"Warning: Previous run '{self._current_run.run_id}' was not ended. Ending it now.")
            self.end_run()
        
        run_id = self._generate_run_id(model_name)
        self._run_start_time = datetime.now()
        
        self._current_run = RunInfo(
            run_id=run_id,
            model_name=model_name,
            timestamp=self._run_start_time.isoformat(),
            params={},
            metrics={},
            artifacts=[],
            notes=notes
        )
        
        # Create run directory
        self._current_run_dir = self.experiment_dir / run_id
        self._current_run_dir.mkdir(parents=True, exist_ok=True)
        (self._current_run_dir / "plots").mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Started run: {run_id}")
        print(f"Model: {model_name}")
        print(f"Output dir: {self._current_run_dir}")
        print(f"{'='*60}")
        
        return run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Zapisz hiperparametry runu"""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        self._current_run.params.update(params)
        
        # Save to file
        params_file = self._current_run_dir / "params.json"
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(self._current_run.params, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Zapisz metryki runu.
        
        Args:
            metrics: Słownik z metrykami
            step: Opcjonalny krok (np. epoch)
        """
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        self._current_run.metrics.update(metrics)
        
        # Save to file
        metrics_file = self._current_run_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self._current_run.metrics, f, indent=2)
        
        # Log step metrics if provided
        if step is not None:
            step_file = self._current_run_dir / "metrics_history.jsonl"
            with open(step_file, 'a', encoding='utf-8') as f:
                entry = {'step': step, 'timestamp': datetime.now().isoformat(), **metrics}
                f.write(json.dumps(entry) + '\n')
    
    def log_artifact(self, source_path: Path, artifact_name: str = None):
        """
        Zapisz artefakt (plik) do katalogu runu.
        
        Args:
            source_path: Ścieżka do pliku źródłowego
            artifact_name: Opcjonalna nazwa artefaktu (default: nazwa pliku)
        """
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        source_path = Path(source_path)
        if not source_path.exists():
            print(f"Warning: Artifact not found: {source_path}")
            return
        
        artifact_name = artifact_name or source_path.name
        dest_path = self._current_run_dir / artifact_name
        
        shutil.copy2(source_path, dest_path)
        self._current_run.artifacts.append(artifact_name)
        
        print(f"  Logged artifact: {artifact_name}")
    
    def log_model(self, model_path: Path):
        """Zapisz model jako artefakt"""
        self.log_artifact(model_path, f"model{model_path.suffix}")
    
    def get_run_dir(self) -> Optional[Path]:
        """Zwróć katalog aktualnego runu"""
        return self._current_run_dir
    
    def end_run(self, status: str = "completed"):
        """
        Zakończ aktualny run.
        
        Args:
            status: Status runu ('completed', 'failed', 'interrupted')
        """
        if self._current_run is None:
            print("Warning: No active run to end.")
            return
        
        # Calculate duration
        if self._run_start_time:
            duration = (datetime.now() - self._run_start_time).total_seconds()
            self._current_run.duration_seconds = duration
        
        # Add to runs list
        self.runs.append(self._current_run)
        self._save_runs()
        
        # Print summary
        print(f"\n{'-'*60}")
        print(f"Run completed: {self._current_run.run_id}")
        print(f"Duration: {self._current_run.duration_seconds:.1f}s")
        print(f"Metrics:")
        for k, v in self._current_run.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"{'-'*60}\n")
        
        # Reset state
        self._current_run = None
        self._current_run_dir = None
        self._run_start_time = None
    
    def log_run(self, model_name: str, params: Dict, metrics: Dict, 
                artifacts: List[Path] = None, notes: str = ""):
        """
        Convenience method - zapisz cały run w jednym wywołaniu.
        
        Args:
            model_name: Nazwa modelu
            params: Hiperparametry
            metrics: Metryki
            artifacts: Lista ścieżek do artefaktów
            notes: Notatki
        """
        self.start_run(model_name, notes)
        self.log_params(params)
        self.log_metrics(metrics)
        
        if artifacts:
            for artifact_path in artifacts:
                self.log_artifact(artifact_path)
        
        self.end_run()
    
    def get_best_run(self, metric: str = 'f1', higher_is_better: bool = True) -> Optional[RunInfo]:
        """
        Zwróć najlepszy run według metryki.
        
        Args:
            metric: Nazwa metryki
            higher_is_better: Czy wyższa wartość jest lepsza
            
        Returns:
            RunInfo najlepszego runu lub None
        """
        if not self.runs:
            return None
        
        valid_runs = [r for r in self.runs if metric in r.metrics]
        if not valid_runs:
            return None
        
        if higher_is_better:
            return max(valid_runs, key=lambda r: r.metrics[metric])
        else:
            return min(valid_runs, key=lambda r: r.metrics[metric])
    
    def get_runs_by_model(self, model_name: str) -> List[RunInfo]:
        """Zwróć wszystkie runy dla danego modelu"""
        return [r for r in self.runs if r.model_name == model_name]
    
    def print_summary(self):
        """Wydrukuj podsumowanie wszystkich runów"""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*80}")
        print(f"Total runs: {len(self.runs)}")
        
        if not self.runs:
            print("No runs recorded yet.")
            return
        
        # Group by model
        models = {}
        for run in self.runs:
            if run.model_name not in models:
                models[run.model_name] = []
            models[run.model_name].append(run)
        
        print(f"\n{'Model':<25} {'Runs':>6} {'Best F1':>10} {'Best AUROC':>12}")
        print("-" * 60)
        
        for model_name, model_runs in models.items():
            n_runs = len(model_runs)
            best_f1 = max((r.metrics.get('f1', 0) for r in model_runs), default=0)
            best_auroc = max((r.metrics.get('auroc', 0) for r in model_runs), default=0)
            
            print(f"{model_name[:24]:<25} {n_runs:>6} {best_f1:>10.4f} {best_auroc:>12.4f}")
        
        print("-" * 60)
        
        # Best overall
        best_run = self.get_best_run('f1')
        if best_run:
            print(f"\n*** Best overall: {best_run.model_name}")
            print(f"   Run ID: {best_run.run_id}")
            print(f"   F1: {best_run.metrics.get('f1', 0):.4f}")
        
        print(f"{'='*80}\n")
    
    def export_comparison_csv(self, output_path: Path = None):
        """Eksportuj porównanie runów do CSV"""
        import pandas as pd
        
        if not self.runs:
            print("No runs to export.")
            return
        
        rows = []
        for run in self.runs:
            row = {
                'run_id': run.run_id,
                'model': run.model_name,
                'timestamp': run.timestamp,
                'duration_s': run.duration_seconds,
                **{f'param_{k}': v for k, v in run.params.items()},
                **run.metrics
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        output_path = output_path or self.experiment_dir / "all_runs.csv"
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} runs to {output_path}")
        
        return df


if __name__ == '__main__':
    # Test tracker
    print("Testing ExperimentTracker...")
    
    tracker = ExperimentTracker("test_experiment")
    
    # Simulate a run
    tracker.start_run("TestModel", notes="Test run")
    tracker.log_params({
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    })
    tracker.log_metrics({'f1': 0.75, 'auroc': 0.82, 'accuracy': 0.80})
    tracker.end_run()
    
    # Print summary
    tracker.print_summary()
    
    # Get best run
    best = tracker.get_best_run('f1')
    if best:
        print(f"Best run: {best.run_id} with F1={best.metrics['f1']:.4f}")
