"""
Final Model Comparison

Compares all trained models and generates publication-ready results:
1. XGBoost baseline
2. MIL Text model
3. Audio CNN model
4. Multimodal Fusion

Outputs:
- Comparison table (CSV)
- LaTeX table for paper
- Summary report
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG
from evaluation.report import EvaluationReport, ModelComparison


def load_all_results() -> dict:
    """Load all result JSON files"""
    results = {}
    
    for json_file in CONFIG.RESULTS_DIR.glob("*_results.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model_name = data.get('model_name', json_file.stem)
            results[model_name] = data
            print(f"Loaded: {model_name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create comparison DataFrame"""
    rows = []
    
    for model_name, data in results.items():
        metrics = data.get('metrics', {})
        ci = data.get('bootstrap_ci', {})
        
        row = {
            'Model': model_name,
            'F1': metrics.get('f1', 0),
            'F1_CI_Lower': ci.get('f1', {}).get('lower_ci', np.nan),
            'F1_CI_Upper': ci.get('f1', {}).get('upper_ci', np.nan),
            'AUROC': metrics.get('auroc', 0),
            'AUROC_CI_Lower': ci.get('auroc', {}).get('lower_ci', np.nan),
            'AUROC_CI_Upper': ci.get('auroc', {}).get('upper_ci', np.nan),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'Accuracy': metrics.get('accuracy', 0),
            'MCC': metrics.get('mcc', 0),
            'Sensitivity': metrics.get('sensitivity', 0),
            'Specificity': metrics.get('specificity', 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('F1', ascending=False)
    
    return df


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper"""
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Model Comparison on DAIC-WOZ Test Set}")
    latex.append(r"\label{tab:model_comparison}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & F1 Score & AUROC & Precision & Recall \\")
    latex.append(r"\midrule")
    
    for _, row in df.iterrows():
        model = row['Model'].replace('_', r'\_')
        
        # Format F1 with CI
        if not np.isnan(row['F1_CI_Lower']):
            f1_str = f"${row['F1']:.3f}$ $[{row['F1_CI_Lower']:.3f}, {row['F1_CI_Upper']:.3f}]$"
        else:
            f1_str = f"${row['F1']:.3f}$"
        
        # Format AUROC with CI
        if not np.isnan(row['AUROC_CI_Lower']):
            auroc_str = f"${row['AUROC']:.3f}$ $[{row['AUROC_CI_Lower']:.3f}, {row['AUROC_CI_Upper']:.3f}]$"
        else:
            auroc_str = f"${row['AUROC']:.3f}$"
        
        latex.append(f"{model} & {f1_str} & {auroc_str} & ${row['Precision']:.3f}$ & ${row['Recall']:.3f}$ \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)


def generate_summary_report(results: dict, df: pd.DataFrame) -> str:
    """Generate comprehensive summary report"""
    lines = []
    
    lines.append("=" * 80)
    lines.append("MULTIMODAL DEPRESSION DETECTION - FINAL COMPARISON REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    
    lines.append("\n## EXECUTIVE SUMMARY\n")
    
    if len(df) > 0:
        best_model = df.iloc[0]['Model']
        best_f1 = df.iloc[0]['F1']
        lines.append(f"Best Model: {best_model}")
        lines.append(f"Best F1 Score: {best_f1:.4f}")
        
        if len(df) > 1:
            improvement = best_f1 - df.iloc[-1]['F1']
            lines.append(f"Improvement over worst: +{improvement:.4f}")
    
    lines.append("\n## DETAILED COMPARISON\n")
    lines.append("-" * 80)
    lines.append(f"{'Model':<35} {'F1':>10} {'AUROC':>10} {'Prec':>10} {'Recall':>10}")
    lines.append("-" * 80)
    
    for _, row in df.iterrows():
        lines.append(f"{row['Model'][:34]:<35} {row['F1']:>10.4f} {row['AUROC']:>10.4f} "
                    f"{row['Precision']:>10.4f} {row['Recall']:>10.4f}")
    
    lines.append("-" * 80)
    
    # Additional clinical metrics
    lines.append("\n## CLINICAL METRICS\n")
    lines.append("-" * 80)
    lines.append(f"{'Model':<35} {'Sensitivity':>12} {'Specificity':>12} {'MCC':>10}")
    lines.append("-" * 80)
    
    for _, row in df.iterrows():
        lines.append(f"{row['Model'][:34]:<35} {row['Sensitivity']:>12.4f} "
                    f"{row['Specificity']:>12.4f} {row['MCC']:>10.4f}")
    
    lines.append("-" * 80)
    
    # Key findings
    lines.append("\n## KEY FINDINGS\n")
    
    # Check for multimodal improvement
    single_modality = df[df['Model'].str.contains('Audio|Text', case=False, regex=True)]
    multimodal = df[df['Model'].str.contains('Multimodal|Fusion', case=False, regex=True)]
    
    if len(single_modality) > 0 and len(multimodal) > 0:
        best_single = single_modality['F1'].max()
        best_multi = multimodal['F1'].max()
        
        if best_multi > best_single:
            lines.append(f"• Multimodal fusion improves F1 by +{best_multi - best_single:.4f} over best single modality")
        else:
            lines.append(f"• Single modality ({single_modality.iloc[0]['Model']}) outperforms multimodal fusion")
    
    # Check MIL improvement
    mil_models = df[df['Model'].str.contains('MIL', case=False)]
    non_mil = df[~df['Model'].str.contains('MIL', case=False)]
    
    if len(mil_models) > 0 and len(non_mil) > 0:
        best_mil = mil_models['F1'].max()
        best_non_mil = non_mil['F1'].max()
        
        if best_mil > best_non_mil:
            lines.append(f"• MIL approach improves F1 by +{best_mil - best_non_mil:.4f}")
    
    # Statistical significance note
    lines.append("\n## STATISTICAL NOTES\n")
    lines.append(f"• Bootstrap CI computed with {CONFIG.BOOTSTRAP_CI_RESAMPLES} resamples")
    lines.append(f"• Confidence level: {CONFIG.CONFIDENCE_LEVEL*100:.0f}%")
    lines.append(f"• Random seed: {CONFIG.RANDOM_SEED}")
    
    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Final Model Comparison")
    print("=" * 60)
    
    # Load results
    print("\nLoading results...")
    results = load_all_results()
    
    if not results:
        print("No results found in results/ directory.")
        print("Run the training scripts first:")
        print("  python experiments/run_baselines.py")
        print("  python experiments/train_mil_text.py")
        print("  python experiments/train_multimodal.py")
        return
    
    print(f"\nFound {len(results)} model results")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Save CSV
    csv_path = CONFIG.RESULTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nComparison table saved to {csv_path}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    latex_path = CONFIG.RESULTS_DIR / "comparison_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_path}")
    
    # Generate summary report
    report = generate_summary_report(results, df)
    report_path = CONFIG.RESULTS_DIR / "final_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Summary report saved to {report_path}")
    
    # Print report
    print("\n" + report)


if __name__ == '__main__':
    main()
