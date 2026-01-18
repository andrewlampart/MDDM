"""Evaluation framework for depression detection models"""
from .metrics import MetricsComputer
from .report import EvaluationReport
from .calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    calibrate_model,
    evaluate_calibration,
    calibrate_and_evaluate,
    plot_calibration_comparison,
    compute_ece,
    compute_brier,
    TemperatureScaling,
)
from .ablation import (
    AblationStudy,
    run_ablation_from_files,
)

__all__ = [
    'MetricsComputer',
    'EvaluationReport',
    # Calibration
    'expected_calibration_error',
    'maximum_calibration_error',
    'calibrate_model',
    'evaluate_calibration',
    'calibrate_and_evaluate',
    'plot_calibration_comparison',
    'compute_ece',
    'compute_brier',
    'TemperatureScaling',
    # Ablation
    'AblationStudy',
    'run_ablation_from_files',
]
