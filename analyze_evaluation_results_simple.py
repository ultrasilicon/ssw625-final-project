#!/usr/bin/env python3
"""
Script to analyze evaluation results and prepare data for box plots
Combines results from all 4 evaluation models for comprehensive analysis
(No matplotlib/seaborn dependencies - pure analysis)
"""

import json
import csv
import statistics
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

class EvaluationResultsAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.models = ["Deepseek-V3-0324", "DeepSeek-V3.1", "Grok-3", "GPT-4.1"]
        self.agents = ["Claude_Code", "Copilot", "Cursor", "Devin", "OpenAI_Codex"]
        self.criteria = [
            "coverage_purpose", "coverage_changes", "rationale_clarity",
            "commit_coverage", "testing_guidance", "readability"
        ]
        self.all_data = []
        
    def load_all_results(self):
        """Load all evaluation results from all models and agents."""
        print("Loading evaluation results...")
        
        for model in self.models:
            model_dir = self.results_dir / model
            if not model_dir.exists():
                print(f"Warning: Model directory {model} not found")
                continue
                
            print(f"  Loading {model} results...")
            
            for agent in self.agents:
                agent_file = model_dir / f"{agent}_output.json"
                if not agent_file.exists():
                    print(f"    Warning: {agent} results not found for {model}")
                    continue
                
                try:
                    with open(agent_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract individual PR results
                    for result in data.get('results', []):
                        self.all_data.append({
                            'model': model,
                            'agent': agent,
                            'pr_id': result['pr_id'],
                            'coverage_purpose': result['coverage_purpose'],
                            'coverage_changes': result['coverage_changes'],
                            'rationale_clarity': result['rationale_clarity'],
                            'commit_coverage': result['commit_coverage'],
                            'testing_guidance': result['testing_guidance'],
                            'readability': result['readability']
                        })
                    
                    print(f"    Loaded {len(data.get('results', []))} PRs for {agent}")
                    
                except Exception as e:
                    print(f"    Error loading {agent} results for {model}: {e}")
        
        print(f"Total loaded: {len(self.all_data)} evaluation records")
        return len(self.all_data) > 0
    
    def calculate_statistics(self, scores):
        """Calculate comprehensive statistics for a list of scores."""
        if not scores:
            return {}
        
        return {
            'count': len(scores),
            'mean': statistics.mean(scores),
            'median': statistics.median(scores),
            'mode': statistics.mode(scores) if len(set(scores)) < len(scores) else scores[0],
            'std': statistics.stdev(scores) if len(scores) > 1 else 0,
            'min': min(scores),
            'max': max(scores),
            'q25': statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else min(scores),
            'q75': statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else max(scores)
        }
    
    def prepare_boxplot_data(self):
        """Prepare data specifically for box plot visualization."""
        print("\nPreparing box plot data...")
        
        boxplot_data = {}
        
        # Organize data by criterion for box plots
        for criterion in self.criteria:
            criterion_data = {agent: [] for agent in self.agents}
            
            # Collect all scores for each agent for this criterion
            for record in self.all_data:
                agent = record['agent']
                score = record[criterion]
                criterion_data[agent].append(score)
            
            # Format for box plot
            boxplot_records = []
            for agent in self.agents:
                scores = criterion_data[agent]
                for score in scores:
                    boxplot_records.append({
                        'Agent': agent,
                        'Score': score,
                        'Criterion': criterion.replace('_', ' ').title()
                    })
            
            boxplot_data[criterion] = {
                'records': boxplot_records,
                'agent_scores': criterion_data,
                'total_points': sum(len(scores) for scores in criterion_data.values())
            }
            
            print(f"  {criterion}: {boxplot_data[criterion]['total_points']} data points")
        
        return boxplot_data
    
    def generate_agent_summary(self):
        """Generate comprehensive summary statistics by agent."""
        print("\nGenerating agent summary statistics...")
        
        agent_summary = {}
        
        for agent in self.agents:
            agent_data = [record for record in self.all_data if record['agent'] == agent]
            
            if not agent_data:
                continue
            
            agent_stats = {
                'agent': agent,
                'total_evaluations': len(agent_data),
                'models_evaluated': len(set(record['model'] for record in agent_data))
            }
            
            # Calculate statistics for each criterion
            for criterion in self.criteria:
                scores = [record[criterion] for record in agent_data]
                stats = self.calculate_statistics(scores)
                
                # Add criterion prefix to stats
                for stat_name, stat_value in stats.items():
                    agent_stats[f'{criterion}_{stat_name}'] = stat_value
            
            # Calculate overall statistics across all criteria
            all_scores = []
            for criterion in self.criteria:
                all_scores.extend([record[criterion] for record in agent_data])
            
            overall_stats = self.calculate_statistics(all_scores)
            for stat_name, stat_value in overall_stats.items():
                agent_stats[f'overall_{stat_name}'] = stat_value
            
            agent_summary[agent] = agent_stats
            print(f"  {agent}: {len(agent_data)} evaluations")
        
        return agent_summary
    
    def generate_model_comparison(self):
        """Generate comparison statistics across evaluation models."""
        print("\nGenerating model comparison...")
        
        model_comparison = {}
        
        for model in self.models:
            model_data = [record for record in self.all_data if record['model'] == model]
            
            if not model_data:
                continue
            
            model_stats = {
                'model': model,
                'total_evaluations': len(model_data),
                'agents_evaluated': len(set(record['agent'] for record in model_data))
            }
            
            # Calculate statistics for each criterion across all agents
            for criterion in self.criteria:
                scores = [record[criterion] for record in model_data]
                stats = self.calculate_statistics(scores)
                
                for stat_name, stat_value in stats.items():
                    model_stats[f'{criterion}_{stat_name}'] = stat_value
            
            # Overall statistics
            all_scores = []
            for criterion in self.criteria:
                all_scores.extend([record[criterion] for record in model_data])
            
            overall_stats = self.calculate_statistics(all_scores)
            for stat_name, stat_value in overall_stats.items():
                model_stats[f'overall_{stat_name}'] = stat_value
            
            model_comparison[model] = model_stats
            print(f"  {model}: {len(model_data)} evaluations")
        
        return model_comparison
    
    def create_criterion_analysis(self):
        """Analyze performance across different criteria."""
        print("\nAnalyzing criteria performance...")
        
        criteria_analysis = {}
        
        for criterion in self.criteria:
            # Get all scores for this criterion across all agents and models
            all_scores = [record[criterion] for record in self.all_data]
            
            # By agent
            agent_scores = {}
            for agent in self.agents:
                agent_data = [record[criterion] for record in self.all_data if record['agent'] == agent]
                if agent_data:
                    agent_scores[agent] = self.calculate_statistics(agent_data)
            
            # By model
            model_scores = {}
            for model in self.models:
                model_data = [record[criterion] for record in self.all_data if record['model'] == model]
                if model_data:
                    model_scores[model] = self.calculate_statistics(model_data)
            
            criteria_analysis[criterion] = {
                'overall': self.calculate_statistics(all_scores),
                'by_agent': agent_scores,
                'by_model': model_scores
            }
            
            print(f"  {criterion}: {len(all_scores)} total scores")
        
        return criteria_analysis
    
    def save_csv_data(self, output_dir):
        """Save data in CSV format for easy analysis."""
        print("\nSaving CSV data files...")
        
        # 1. Raw combined data
        csv_file = output_dir / "combined_evaluation_data.csv"
        with open(csv_file, 'w', newline='') as f:
            if self.all_data:
                fieldnames = self.all_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.all_data)
        print(f"  Saved: {csv_file}")
        
        # 2. Box plot data for each criterion
        boxplot_data = self.prepare_boxplot_data()
        for criterion, data in boxplot_data.items():
            csv_file = output_dir / f"boxplot_{criterion}.csv"
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ['Agent', 'Score', 'Criterion']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data['records'])
            print(f"  Saved: {csv_file}")
        
        return boxplot_data
    
    def save_summary_report(self, output_dir, agent_summary, model_comparison, criteria_analysis):
        """Save a comprehensive text report."""
        report_file = output_dir / "evaluation_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION RESULTS COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Dataset overview
            f.write(f"Dataset Overview:\n")
            f.write(f"  Total Records: {len(self.all_data):,}\n")
            f.write(f"  Models: {len(self.models)} ({', '.join(self.models)})\n")
            f.write(f"  Agents: {len(self.agents)} ({', '.join(self.agents)})\n")
            f.write(f"  Criteria: {len(self.criteria)}\n\n")
            
            # Agent performance ranking
            f.write("Agent Performance Ranking (by Overall Mean Score):\n")
            f.write("-" * 50 + "\n")
            agent_rankings = []
            for agent, stats in agent_summary.items():
                agent_rankings.append((agent, stats['overall_mean']))
            agent_rankings.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (agent, score) in enumerate(agent_rankings, 1):
                f.write(f"  {rank}. {agent:<15}: {score:.3f}\n")
            f.write("\n")
            
            # Model comparison
            f.write("Model Performance Comparison (by Overall Mean Score):\n")
            f.write("-" * 50 + "\n")
            model_rankings = []
            for model, stats in model_comparison.items():
                model_rankings.append((model, stats['overall_mean']))
            model_rankings.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (model, score) in enumerate(model_rankings, 1):
                f.write(f"  {rank}. {model:<20}: {score:.3f}\n")
            f.write("\n")
            
            # Criteria analysis
            f.write("Criteria Performance Summary:\n")
            f.write("-" * 50 + "\n")
            for criterion, analysis in criteria_analysis.items():
                overall_stats = analysis['overall']
                f.write(f"  {criterion.replace('_', ' ').title()}:\n")
                f.write(f"    Mean: {overall_stats['mean']:.3f} ± {overall_stats['std']:.3f}\n")
                f.write(f"    Range: {overall_stats['min']} - {overall_stats['max']}\n")
                f.write(f"    Median: {overall_stats['median']:.3f}\n\n")
            
            # Detailed agent statistics
            f.write("\nDetailed Agent Statistics:\n")
            f.write("="*50 + "\n")
            for agent, stats in agent_summary.items():
                f.write(f"\n{agent}:\n")
                f.write(f"  Evaluations: {stats['total_evaluations']}\n")
                f.write(f"  Models: {stats['models_evaluated']}\n")
                f.write(f"  Overall: {stats['overall_mean']:.3f} ± {stats['overall_std']:.3f}\n")
                
                f.write(f"  Criteria breakdown:\n")
                for criterion in self.criteria:
                    mean_key = f'{criterion}_mean'
                    std_key = f'{criterion}_std'
                    if mean_key in stats:
                        f.write(f"    {criterion.replace('_', ' ').title()}: {stats[mean_key]:.3f} ± {stats[std_key]:.3f}\n")
        
        print(f"  Saved: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive evaluation results analysis...")
        print("="*80)
        
        # Load data
        if not self.load_all_results():
            print("Failed to load data. Exiting.")
            return None
        
        # Create output directory
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        # Generate analyses
        agent_summary = self.generate_agent_summary()
        model_comparison = self.generate_model_comparison()
        criteria_analysis = self.create_criterion_analysis()
        
        # Save data
        boxplot_data = self.save_csv_data(output_dir)
        
        # Save JSON results
        json_results = {
            'agent_summary': agent_summary,
            'model_comparison': model_comparison,
            'criteria_analysis': criteria_analysis,
            'boxplot_data': {k: v['agent_scores'] for k, v in boxplot_data.items()}
        }
        
        json_file = output_dir / "complete_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"  Saved: {json_file}")
        
        # Save summary report
        report_file = self.save_summary_report(output_dir, agent_summary, model_comparison, criteria_analysis)
        
        # Print summary to console
        self.print_console_summary(agent_summary, model_comparison, criteria_analysis)
        
        return {
            'agent_summary': agent_summary,
            'model_comparison': model_comparison,
            'criteria_analysis': criteria_analysis,
            'boxplot_data': boxplot_data,
            'output_directory': output_dir
        }
    
    def print_console_summary(self, agent_summary, model_comparison, criteria_analysis):
        """Print key findings to console."""
        print("\n" + "="*80)
        print("KEY FINDINGS SUMMARY")
        print("="*80)
        
        print(f"\nTop 3 Performing Agents (Overall Mean):")
        agent_rankings = [(agent, stats['overall_mean']) for agent, stats in agent_summary.items()]
        agent_rankings.sort(key=lambda x: x[1], reverse=True)
        for i, (agent, score) in enumerate(agent_rankings[:3], 1):
            print(f"  {i}. {agent}: {score:.3f}")
        
        print(f"\nTop 3 Performing Models (Overall Mean):")
        model_rankings = [(model, stats['overall_mean']) for model, stats in model_comparison.items()]
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(model_rankings[:3], 1):
            print(f"  {i}. {model}: {score:.3f}")
        
        print(f"\nCriteria Ranking (Highest to Lowest Average Score):")
        criteria_rankings = []
        for criterion, analysis in criteria_analysis.items():
            criteria_rankings.append((criterion, analysis['overall']['mean']))
        criteria_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for i, (criterion, score) in enumerate(criteria_rankings, 1):
            print(f"  {i}. {criterion.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\nBox Plot Data Ready:")
        print(f"  • {len(self.criteria)} criteria datasets created")
        print(f"  • Each dataset contains scores for all {len(self.agents)} agents")
        print(f"  • Use files: boxplot_[criterion].csv")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    analyzer = EvaluationResultsAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        output_dir = results['output_directory']
        print(f"\n🎉 Analysis Complete!")
        print(f"\nFiles created in '{output_dir}':")
        print(f"  📊 Box Plot Data:")
        for criterion in analyzer.criteria:
            print(f"    • boxplot_{criterion}.csv")
        print(f"  📈 Analysis Files:")
        print(f"    • combined_evaluation_data.csv - Raw data")
        print(f"    • complete_analysis.json - Full analysis results")
        print(f"    • evaluation_summary_report.txt - Human-readable report")
        
        print(f"\n📋 Quick Usage for Box Plots:")
        print(f"  import pandas as pd")
        print(f"  import matplotlib.pyplot as plt")
        print(f"  import seaborn as sns")
        print(f"")
        print(f"  # Example for coverage_purpose criterion:")
        print(f"  data = pd.read_csv('analysis_results/boxplot_coverage_purpose.csv')")
        print(f"  sns.boxplot(data=data, x='Agent', y='Score')")
        print(f"  plt.title('Coverage Purpose Scores by Agent')")
        print(f"  plt.xticks(rotation=45)")
        print(f"  plt.show()")
        
    else:
        print("❌ Analysis failed. Please check the results directory structure.")

if __name__ == "__main__":
    main()