import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import json
from sklearn.metrics import roc_auc_score
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
import argparse
from pprint import pprint
from loguru import logger
import nltk
import pandas as pd
from pathlib import Path

nltk.download('punkt')


class MetricsCalculator:
    """Handles all metric calculations for evaluation."""
    
    @staticmethod
    def calculate_accuracy(df: pd.DataFrame) -> float:
        """Calculate accuracy metric."""
        if df.empty:
            return 0.0
        return (df['gt-label'] == df['prediction']).mean()
    
    @staticmethod
    def calculate_precision(df: pd.DataFrame) -> float:
        """Calculate precision metric."""
        if df.empty:
            return 0.0
        true_positives = len(df[(df['prediction'] == True) & (df['gt-label'] == True)])
        predicted_positives = len(df[df['prediction'] == True])
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    @staticmethod
    def calculate_recall(df: pd.DataFrame) -> float:
        """Calculate recall metric."""
        if df.empty:
            return 0.0
        true_positives = len(df[(df['prediction'] == True) & (df['gt-label'] == True)])
        actual_positives = len(df[df['gt-label'] == True])
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    @staticmethod
    def calculate_roc_auc(df: pd.DataFrame) -> float:
        """Calculate ROC AUC score."""
        if df.empty or len(df['gt-label'].unique()) <= 1:
            return 0.0
        return roc_auc_score(df['gt-label'].astype(int), df['q-score'])
    
    @staticmethod
    def calculate_bleu_score(df: pd.DataFrame) -> float:
        """Calculate BLEU score for backdoored models."""
        if df.empty:
            return 0.0
        
        backdoored_df = df[df['gt-label'] == True]
        if backdoored_df.empty:
            return 0.0
        
        def compute_bleu(row):
            return sentence_bleu(
                [word_tokenize(str(row['gt-target']).lower())],
                word_tokenize(str(row['invert-target']).lower())
            )
        
        return backdoored_df.apply(compute_bleu, axis=1).mean()
    
    @staticmethod
    def calculate_overhead(df: pd.DataFrame) -> float:
        """Calculate average time overhead."""
        return df['time-taken'].mean() if not df.empty else 0.0


class ErrorAnalyzer:
    """Handles error analysis for evaluation results."""
    
    @staticmethod
    def get_error_cases(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get false positives and false negatives."""
        false_positives = df[
            (df['prediction'] == True) & (df['gt-label'] == False)
        ]['model-id'].tolist()
        
        false_negatives = df[
            (df['prediction'] == False) & (df['gt-label'] == True)
        ]['model-id'].tolist()
        
        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }


class ReportGenerator:
    """Handles report generation and file output."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.metrics_calc = MetricsCalculator()
        self.error_analyzer = ErrorAnalyzer()
    
    def generate_metrics_report(self, df: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Generate comprehensive metrics report."""
        if df is None:
            df = self.df
        
        if df.empty:
            return self._get_empty_report()
        
        precision = self.metrics_calc.calculate_precision(df)
        recall = self.metrics_calc.calculate_recall(df)
        
        report = {
            "accuracy": self.metrics_calc.calculate_accuracy(df),
            "precision": precision,
            "recall": recall,
            "f1_score": self.metrics_calc.calculate_f1_score(precision, recall),
            "roc_auc_score": self.metrics_calc.calculate_roc_auc(df),
            "bleu_score": self.metrics_calc.calculate_bleu_score(df),
            "overhead": self.metrics_calc.calculate_overhead(df),
        }
        
        error_analysis = self.error_analyzer.get_error_cases(df)
        
        return report, error_analysis
    
    def _get_empty_report(self) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Return empty report structure."""
        report = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc_score": 0.0,
            "bleu_score": 0.0,
            "overhead": 0.0,
        }
        
        error_analysis = {
            "false_positives": [],
            "false_negatives": []
        }
        
        return report, error_analysis
    
    def save_to_markdown(self, filepath: str):
        """Save evaluation results to markdown file."""
        with open(filepath, 'w') as f:
            self._write_header(f)
            self._write_summary(f)
            self._write_results_table(f)
            self._write_error_analysis(f)
            self._write_error_cases_table(f)
    
    def _write_header(self, f):
        """Write markdown header."""
        f.write("# BAIT Evaluation Results\n\n")
    
    def _write_summary(self, f):
        """Write summary statistics."""
        f.write("## Summary\n")
        f.write(f"- Total models evaluated: {len(self.df)}\n")
        f.write("- Model types:\n")
        for model_type in self.df['model-type'].dropna().unique():
            f.write(f"  - {model_type}\n")
        f.write("- Datasets:\n")
        for dataset in self.df['dataset'].dropna().unique():
            f.write(f"  - {dataset}\n")
        f.write("\n")
    
    def _write_results_table(self, f):
        """Write results table by model type and dataset."""
        f.write("## Results by Model Type and Dataset\n\n")
        f.write("| Dataset | # Models | Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC | BLEU | Overhead |\n")
        f.write("|---------|--------------|------------|----------|-----------|--------|----------|---------|------|----------|\n")
        
        # Write per-combination results
        for model_type in self.df['model-type'].unique():
            for dataset in self.df['dataset'].unique():
                subset_df = self._select_data(dataset, model_type)
                if not subset_df.empty:
                    report, _ = self.generate_metrics_report(subset_df)
                    self._write_results_row(f, dataset, len(subset_df), model_type, report)
        
        # Write overall results
        overall_report, _ = self.generate_metrics_report()
        self._write_results_row(f, "**All**", len(self.df), "**All**", overall_report, bold=True)
        f.write("\n")
    
    def _write_results_row(self, f, dataset: str, total_models: int, model_type: str, report: Dict[str, float], bold: bool = False):
        """Write a single results row to the table."""
        format_func = lambda x: f"**{x:.3f}**" if bold else f"{x:.3f}"
        total_format = f"**{total_models}**" if bold else str(total_models)
        f.write(f"| {dataset} | {total_format} | {model_type} | "
               f"{format_func(report['accuracy'])} | {format_func(report['precision'])} | "
               f"{format_func(report['recall'])} | {format_func(report['f1_score'])} | "
               f"{format_func(report['roc_auc_score'])} | {format_func(report['bleu_score'])} | "
               f"{format_func(report['overhead'])} |\n")
    
    def _write_error_analysis(self, f):
        """Write error analysis section."""
        _, errors = self.generate_metrics_report()
        
        f.write("## Error Analysis\n\n")
        f.write(f"### False Positives ({len(errors['false_positives'])} models)\n")
        for fp in errors['false_positives']:
            f.write(f"- {fp}\n")
        
        f.write(f"\n### False Negatives ({len(errors['false_negatives'])} models)\n")
        for fn in errors['false_negatives']:
            f.write(f"- {fn}\n")
    
    def _write_error_cases_table(self, f):
        """Write detailed error cases table."""
        _, errors = self.generate_metrics_report()
        
        if not errors['false_positives'] and not errors['false_negatives']:
            return
        
        f.write("## Error Cases\n\n")
        f.write("| Type | Model ID | Model Type | Dataset | GT Target | Inverted Target |\n")
        f.write("|------|----------|------------|---------|-----------|----------------|\n")
        
        # Write false positives
        for fp in errors['false_positives']:
            self._write_error_case_row(f, "False Positive", fp)
        
        # Write false negatives
        for fn in errors['false_negatives']:
            self._write_error_case_row(f, "False Negative", fn)
    
    def _write_error_case_row(self, f, error_type: str, model_id: str):
        """Write a single error case row."""
        row = self.df[self.df['model-id'] == model_id].iloc[0]
        gt_target = str(row['gt-target']).replace('\n', ' ')
        invert_target = str(row['invert-target']).replace('\n', ' ')
        f.write(f"| {error_type} | {row['model-id']} | {row['model-type']} | "
               f"{row['dataset']} | {gt_target} | {invert_target} |\n")
    
    def _select_data(self, dataset: str, model_type: str) -> pd.DataFrame:
        """Select data from dataframe based on dataset and model type."""
        return self.df[(self.df['dataset'] == dataset) & (self.df['model-type'] == model_type)]


class DataLoader:
    """Handles loading and parsing of evaluation data."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
    
    def load_results(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load results from run directory."""
        df = pd.DataFrame(columns=[
            'model-id', 'dataset', 'model-type', 'gt-label', 'gt-target', 
            'prediction', 'q-score', 'invert-target', 'time-taken'
        ])
        missing_results = []
        
        for model_id in os.listdir(self.run_dir):
            model_path = self.run_dir / model_id
            if model_path.is_dir():
                try:
                    row_data = self._load_model_result(model_path, model_id)
                    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error loading result for {model_id}: {e}")
                    missing_results.append(model_id)
        
        return df, missing_results
    
    def _load_model_result(self, model_path: Path, model_id: str) -> Dict[str, Any]:
        """Load result for a single model."""
        # Load arguments
        args_path = model_path / "arguments.json"
        with open(args_path, 'r') as f:
            args = json.load(f)
        
        # Load results
        result_path = model_path / "result.json"
        with open(result_path, 'r') as f:
            output = json.load(f)
        
        return {
            "model-id": model_id,
            "dataset": args["data_args"]["dataset"],
            "model-type": args["model_args"]["base_model"],
            "gt-label": bool(args["model_args"]["is_backdoor"]),
            "gt-target": args["model_args"]["target"],
            "prediction": bool(output["is_backdoor"]),
            "q-score": output["q_score"],
            "invert-target": output["invert_target"],
            "time-taken": output["time_taken"]
        }


class Evaluator:
    """Main evaluator class that orchestrates the evaluation process."""
    
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.results_file = self.run_dir / 'results.md'
        self.data_loader = DataLoader(self.run_dir)
        self.df = pd.DataFrame()
        self.report_generator = None
    
    def eval(self):
        """Main evaluation method."""
        # Load data
        self.df, missing_results = self.data_loader.load_results()
        
        # Log results
        logger.info(f"Missing results for {len(missing_results)} models: {missing_results}")
        logger.info(f"Evaluating BAIT results for {len(self.df)} models from {self.run_dir}...")
        
        # Generate and save report
        self.report_generator = ReportGenerator(self.df)
        self.report_generator.save_to_markdown(self.results_file)
        logger.info(f"Results saved to {self.results_file}")
    
    def generate_report(self, df: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Generate report using the report generator."""
        if self.report_generator is None:
            self.report_generator = ReportGenerator(self.df)
        return self.report_generator.generate_metrics_report(df)
    
    def save_to_markdown(self, filepath: str = "test.md"):
        """Save results to markdown file."""
        if self.report_generator is None:
            self.report_generator = ReportGenerator(self.df)
        self.report_generator.save_to_markdown(filepath)


