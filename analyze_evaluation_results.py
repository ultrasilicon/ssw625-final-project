#!/usr/bin/env python3
"""
Script to analyze evaluation results and prepare data for box plots
Combines results from all 4 evaluation models for comprehensive analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    def create_dataframe(self):
        """Convert loaded data to pandas DataFrame for analysis."""
        if not self.all_data:
            print("No data loaded. Run load_all_results() first.")
            return None
        
        df = pd.DataFrame(self.all_data)
        print(f"Created DataFrame with shape: {df.shape}")
        return df
    
    def prepare_boxplot_data(self, df):
        """Prepare data specifically for box plot visualization."""
        boxplot_data = {}
        
        print("\nPreparing box plot data...")
        
        for criterion in self.criteria:
            criterion_data = []
            
            for agent in self.agents:
                agent_scores = df[df['agent'] == agent][criterion].tolist()
                
                # Create records for box plot
                for score in agent_scores:
                    criterion_data.append({
                        'Agent': agent,
                        'Score': score,
                        'Criterion': criterion.replace('_', ' ').title()
                    })
            
            boxplot_data[criterion] = criterion_data
            print(f"  {criterion}: {len(criterion_data)} data points")
        
        return boxplot_data
    
    def generate_summary_statistics(self, df):
        """Generate comprehensive summary statistics."""
        print("\nGenerating summary statistics...")
        
        # Overall statistics by agent
        agent_summary = []
        for agent in self.agents:
            agent_data = df[df['agent'] == agent]
            if len(agent_data) == 0:
                continue
                
            agent_stats = {
                'Agent': agent,
                'Total_Evaluations': len(agent_data),
                'Models_Evaluated': agent_data['model'].nunique()
            }
            
            # Calculate statistics for each criterion
            for criterion in self.criteria:
                scores = agent_data[criterion]
                agent_stats.update({
                    f'{criterion}_mean': scores.mean(),
                    f'{criterion}_std': scores.std(),
                    f'{criterion}_median': scores.median(),
                    f'{criterion}_q25': scores.quantile(0.25),
                    f'{criterion}_q75': scores.quantile(0.75),
                    f'{criterion}_min': scores.min(),
                    f'{criterion}_max': scores.max()
                })
            
            # Overall average across all criteria
            all_scores = []
            for criterion in self.criteria:
                all_scores.extend(agent_data[criterion].tolist())
            agent_stats['overall_mean'] = np.mean(all_scores)
            agent_stats['overall_std'] = np.std(all_scores)
            
            agent_summary.append(agent_stats)
        
        return pd.DataFrame(agent_summary)
    
    def generate_model_comparison(self, df):
        """Generate comparison statistics across different evaluation models."""
        print("\nGenerating model comparison...")
        
        model_comparison = []
        
        for model in self.models:
            model_data = df[df['model'] == model]
            if len(model_data) == 0:
                continue
            
            model_stats = {
                'Model': model,
                'Total_Evaluations': len(model_data),
                'Agents_Evaluated': model_data['agent'].nunique()
            }
            
            # Calculate average scores across all agents for this model
            for criterion in self.criteria:
                scores = model_data[criterion]
                model_stats.update({
                    f'{criterion}_mean': scores.mean(),
                    f'{criterion}_std': scores.std()
                })
            
            # Overall average
            all_scores = []
            for criterion in self.criteria:
                all_scores.extend(model_data[criterion].tolist())
            model_stats['overall_mean'] = np.mean(all_scores)
            
            model_comparison.append(model_stats)
        
        return pd.DataFrame(model_comparison)
    
    def create_visualization_data(self, df):
        """Create data structures optimized for different types of visualizations."""
        viz_data = {}
        
        # Data for box plots by agent and criterion
        viz_data['boxplot_by_criterion'] = {}
        for criterion in self.criteria:
            criterion_data = []
            for agent in self.agents:
                agent_scores = df[df['agent'] == agent][criterion].tolist()
                criterion_data.extend([{
                    'Agent': agent,
                    'Score': score
                } for score in agent_scores])
            viz_data['boxplot_by_criterion'][criterion] = pd.DataFrame(criterion_data)
        
        # Data for heatmap (agent vs criterion averages)
        heatmap_data = []
        for agent in self.agents:
            agent_data = df[df['agent'] == agent]
            if len(agent_data) == 0:
                continue
            
            agent_row = {'Agent': agent}
            for criterion in self.criteria:
                agent_row[criterion.replace('_', ' ').title()] = agent_data[criterion].mean()
            heatmap_data.append(agent_row)
        
        viz_data['heatmap'] = pd.DataFrame(heatmap_data)
        
        # Data for model comparison
        model_comp_data = []
        for model in self.models:
            for agent in self.agents:
                agent_model_data = df[(df['model'] == model) & (df['agent'] == agent)]
                if len(agent_model_data) == 0:
                    continue
                
                for criterion in self.criteria:
                    scores = agent_model_data[criterion].tolist()
                    for score in scores:
                        model_comp_data.append({
                            'Model': model,
                            'Agent': agent,
                            'Criterion': criterion.replace('_', ' ').title(),
                            'Score': score
                        })
        
        viz_data['model_comparison'] = pd.DataFrame(model_comp_data)
        
        return viz_data
    
    def save_analysis_results(self, df, summary_stats, model_comparison, viz_data, boxplot_data):
        """Save all analysis results to files."""
        print("\nSaving analysis results...")
        
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save raw combined data
        df.to_csv(output_dir / "combined_evaluation_data.csv", index=False)
        print(f"  Saved combined data: {output_dir / 'combined_evaluation_data.csv'}")
        
        # Save summary statistics
        summary_stats.to_csv(output_dir / "agent_summary_statistics.csv", index=False)
        print(f"  Saved agent summary: {output_dir / 'agent_summary_statistics.csv'}")
        
        # Save model comparison
        model_comparison.to_csv(output_dir / "model_comparison.csv", index=False)
        print(f"  Saved model comparison: {output_dir / 'model_comparison.csv'}")
        
        # Save visualization data
        for criterion, data in viz_data['boxplot_by_criterion'].items():
            filename = f"boxplot_data_{criterion}.csv"
            data.to_csv(output_dir / filename, index=False)
            print(f"  Saved boxplot data: {output_dir / filename}")
        
        viz_data['heatmap'].to_csv(output_dir / "heatmap_data.csv", index=False)
        viz_data['model_comparison'].to_csv(output_dir / "model_comparison_detailed.csv", index=False)
        
        # Save boxplot data as JSON for easy loading
        with open(output_dir / "boxplot_data_all.json", 'w') as f:
            json.dump(boxplot_data, f, indent=2)
        print(f"  Saved complete boxplot data: {output_dir / 'boxplot_data_all.json'}")
        
        return output_dir
    
    def print_summary(self, df, summary_stats, model_comparison):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nDataset Overview:")
        print(f"  Total Evaluations: {len(df):,}")
        print(f"  Models: {df['model'].nunique()} ({', '.join(df['model'].unique())})")
        print(f"  Agents: {df['agent'].nunique()} ({', '.join(df['agent'].unique())})")
        print(f"  Unique PRs: {df['pr_id'].nunique():,}")
        
        print(f"\nEvaluation Criteria:")
        for i, criterion in enumerate(self.criteria, 1):
            print(f"  {i}. {criterion.replace('_', ' ').title()}")
        
        print(f"\nAgent Performance Summary (Overall Mean Scores):")
        print("-" * 50)
        for _, row in summary_stats.iterrows():
            print(f"  {row['Agent']:<15}: {row['overall_mean']:.3f} ± {row['overall_std']:.3f}")
        
        print(f"\nModel Performance Summary (Overall Mean Scores):")
        print("-" * 50)
        for _, row in model_comparison.iterrows():
            print(f"  {row['Model']:<20}: {row['overall_mean']:.3f}")
        
        print(f"\nCriteria Performance Summary (Across All Agents & Models):")
        print("-" * 60)
        for criterion in self.criteria:
            scores = df[criterion]
            print(f"  {criterion.replace('_', ' ').title():<20}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        print(f"\n" + "="*80)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive evaluation results analysis...")
        
        # Load data
        if not self.load_all_results():
            print("Failed to load data. Exiting.")
            return None
        
        # Create DataFrame
        df = self.create_dataframe()
        if df is None:
            return None
        
        # Generate analyses
        summary_stats = self.generate_summary_statistics(df)
        model_comparison = self.generate_model_comparison(df)
        viz_data = self.create_visualization_data(df)
        boxplot_data = self.prepare_boxplot_data(df)
        
        # Save results
        output_dir = self.save_analysis_results(
            df, summary_stats, model_comparison, viz_data, boxplot_data
        )
        
        # Print summary
        self.print_summary(df, summary_stats, model_comparison)
        
        return {
            'dataframe': df,
            'summary_statistics': summary_stats,
            'model_comparison': model_comparison,
            'visualization_data': viz_data,
            'boxplot_data': boxplot_data,
            'output_directory': output_dir
        }

def main():
    """Main execution function."""
    analyzer = EvaluationResultsAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        print(f"\nAnalysis complete! Check the '{results['output_directory']}' directory for:")
        print("  • combined_evaluation_data.csv - Raw data for all evaluations")
        print("  • agent_summary_statistics.csv - Summary stats by agent")
        print("  • model_comparison.csv - Comparison across evaluation models")
        print("  • boxplot_data_*.csv - Data ready for box plot visualization")
        print("  • boxplot_data_all.json - Complete box plot data in JSON format")
        print("  • heatmap_data.csv - Agent vs criteria heatmap data")
        
        print(f"\nBox Plot Data Structure:")
        print("  Each criterion has data with columns: ['Agent', 'Score']")
        print("  Use this data to create box plots comparing agents for each criterion")
        
        print(f"\nNext Steps:")
        print("  1. Use the CSV files with matplotlib/seaborn for box plots")
        print("  2. Create comparative visualizations across agents and models")
        print("  3. Analyze performance patterns and outliers")
    else:
        print("Analysis failed. Please check the results directory structure.")

if __name__ == "__main__":
    main()