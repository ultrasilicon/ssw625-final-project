#!/usr/bin/env python3
"""
Script to analyze PR evaluation results and generate box plot statistics with statistical significance testing
Combines data from all 4 evaluation models for each agent and score criteria
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations

class ResultsAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.models = ["DeepSeek-V3.1", "Deepseek-V3-0324", "GPT-4.1", "Grok-3"]
        self.agents = ["Claude_Code", "Copilot", "Cursor", "Devin", "OpenAI_Codex"]
        self.score_criteria = [
            "coverage_purpose", "coverage_changes", "rationale_clarity",
            "commit_coverage", "testing_guidance", "readability"
        ]
        
    def load_all_results(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load all results from all models and agents."""
        all_results = {}
        
        for model in self.models:
            model_dir = self.results_dir / model
            if not model_dir.exists():
                print(f"Warning: Model directory {model} not found, skipping...")
                continue
                
            all_results[model] = {}
            
            for agent in self.agents:
                output_file = model_dir / f"{agent}_output.json"
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                        all_results[model][agent] = data['results']
                        print(f"Loaded {len(data['results'])} results for {model} - {agent}")
                else:
                    print(f"Warning: Output file for {model} - {agent} not found")
                    
        return all_results
    
    def combine_scores_by_agent(self, all_results: Dict) -> Dict[str, Dict[str, List[float]]]:
        """Combine scores from all models for each agent."""
        combined_scores = {}
        
        for agent in self.agents:
            combined_scores[agent] = {criteria: [] for criteria in self.score_criteria}
            
            # Collect scores from all models for this agent
            for model in self.models:
                if model in all_results and agent in all_results[model]:
                    results = all_results[model][agent]
                    
                    for result in results:
                        for criteria in self.score_criteria:
                            if criteria in result:
                                combined_scores[agent][criteria].append(result[criteria])
            
            # Print summary for this agent
            total_scores = sum(len(scores) for scores in combined_scores[agent].values())
            if total_scores > 0:
                print(f"{agent}: {total_scores // len(self.score_criteria)} evaluations collected")
            else:
                print(f"Warning: No scores found for {agent}")
                
        return combined_scores
    
    def calculate_box_plot_stats(self, scores: List[float]) -> Dict[str, float]:
        """Calculate box plot statistics for a list of scores."""
        if not scores:
            return {
                'min': 0, 'max': 0, 'q25': 0, 'median': 0, 'q75': 0,
                'mean': 0, 'std': 0, 'count': 0
            }
        
        scores_array = np.array(scores)
        
        return {
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'q25': float(np.percentile(scores_array, 25)),
            'median': float(np.median(scores_array)),
            'q75': float(np.percentile(scores_array, 75)),
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'count': len(scores)
        }
    
    def calculate_pairwise_p_values(self, combined_scores: Dict) -> Dict:
        """Calculate p-values for pairwise comparisons between agents for each criteria."""
        p_values = {}
        
        for criteria in self.score_criteria:
            p_values[criteria] = {}
            
            # Get all agent pairs for this criteria
            agent_pairs = list(combinations(self.agents, 2))
            
            for agent1, agent2 in agent_pairs:
                if (agent1 in combined_scores and agent2 in combined_scores and
                    criteria in combined_scores[agent1] and criteria in combined_scores[agent2]):
                    
                    scores1 = combined_scores[agent1][criteria]
                    scores2 = combined_scores[agent2][criteria]
                    
                    if len(scores1) > 0 and len(scores2) > 0:
                        # Perform Mann-Whitney U test (non-parametric)
                        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                        
                        # Also perform t-test for comparison
                        t_statistic, t_p_value = stats.ttest_ind(scores1, scores2)
                        
                        pair_key = f"{agent1}_vs_{agent2}"
                        p_values[criteria][pair_key] = {
                            'mann_whitney_u_statistic': float(statistic),
                            'mann_whitney_p_value': float(p_value),
                            't_test_statistic': float(t_statistic),
                            't_test_p_value': float(t_p_value),
                            'significant_at_0.05': p_value < 0.05,
                            'significant_at_0.01': p_value < 0.01,
                            'mean_diff': float(np.mean(scores1) - np.mean(scores2)),
                            'effect_size_cohens_d': self.calculate_cohens_d(scores1, scores2)
                        }
        
        return p_values
    
    def calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def calculate_anova_p_values(self, combined_scores: Dict) -> Dict:
        """Calculate ANOVA p-values to test overall differences between agents for each criteria."""
        anova_results = {}
        
        for criteria in self.score_criteria:
            # Collect scores for all agents for this criteria
            groups = []
            agent_names = []
            
            for agent in self.agents:
                if agent in combined_scores and criteria in combined_scores[agent]:
                    scores = combined_scores[agent][criteria]
                    if len(scores) > 0:
                        groups.append(scores)
                        agent_names.append(agent)
            
            if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                try:
                    f_statistic, p_value = stats.f_oneway(*groups)
                    
                    anova_results[criteria] = {
                        'f_statistic': float(f_statistic),
                        'p_value': float(p_value),
                        'significant_at_0.05': p_value < 0.05,
                        'significant_at_0.01': p_value < 0.01,
                        'agents_compared': agent_names,
                        'num_groups': len(groups)
                    }
                except Exception as e:
                    print(f"Error calculating ANOVA for {criteria}: {e}")
                    anova_results[criteria] = None
        
        return anova_results
    
    def generate_box_plot_summary(self, combined_scores: Dict) -> Dict:
        """Generate box plot statistics and p-values for all agents and criteria."""
        box_plot_stats = {}
        
        for agent in self.agents:
            if agent in combined_scores:
                box_plot_stats[agent] = {}
                
                for criteria in self.score_criteria:
                    scores = combined_scores[agent][criteria]
                    stats = self.calculate_box_plot_stats(scores)
                    box_plot_stats[agent][criteria] = stats
        
        # Calculate statistical significance tests
        print("🧮 Calculating statistical significance tests...")
        pairwise_p_values = self.calculate_pairwise_p_values(combined_scores)
        anova_results = self.calculate_anova_p_values(combined_scores)
        
        # Add statistical results to the summary
        statistical_results = {
            'pairwise_comparisons': pairwise_p_values,
            'anova_results': anova_results,
            'description': {
                'mann_whitney_u': 'Non-parametric test comparing two independent samples',
                't_test': 'Parametric test comparing means of two groups',
                'anova': 'Test for overall differences between all agent groups',
                'cohens_d_interpretation': {
                    'small': '0.2-0.5',
                    'medium': '0.5-0.8', 
                    'large': '>0.8'
                }
            }
        }
        
        return {
            'box_plot_statistics': box_plot_stats,
            'statistical_tests': statistical_results
        }
    
    def print_box_plot_summary(self, analysis_results: Dict):
        """Print formatted box plot statistics and p-values."""
        box_plot_stats = analysis_results['box_plot_statistics']
        statistical_tests = analysis_results['statistical_tests']
        
        print("\n" + "="*80)
        print("BOX PLOT STATISTICS SUMMARY")
        print("="*80)
        print("Data Format: Min | Q25 | Median | Q75 | Max | Mean | Std | Count")
        print("-"*80)
        
        for criteria in self.score_criteria:
            print(f"\n📊 {criteria.upper().replace('_', ' ')}")
            print("-" * 60)
            
            for agent in self.agents:
                if agent in box_plot_stats and criteria in box_plot_stats[agent]:
                    stats = box_plot_stats[agent][criteria]
                    
                    print(f"{agent:<15}: "
                          f"{stats['min']:4.1f} | "
                          f"{stats['q25']:4.1f} | "
                          f"{stats['median']:4.1f} | "
                          f"{stats['q75']:4.1f} | "
                          f"{stats['max']:4.1f} | "
                          f"{stats['mean']:4.1f} | "
                          f"{stats['std']:4.1f} | "
                          f"{stats['count']:4d}")
        
        # Print statistical significance results
        print(f"\n{'='*80}")
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        # Print ANOVA results
        anova_results = statistical_tests['anova_results']
        print("\n🔬 ANOVA Results (Overall differences between agents):")
        print("-" * 60)
        print("Criteria                | F-stat | p-value  | Significant?")
        print("-" * 60)
        
        for criteria in self.score_criteria:
            if criteria in anova_results and anova_results[criteria]:
                result = anova_results[criteria]
                sig_symbol = "***" if result['significant_at_0.01'] else ("**" if result['significant_at_0.05'] else "")
                print(f"{criteria:<22} | {result['f_statistic']:6.2f} | {result['p_value']:8.6f} | {sig_symbol}")
        
        # Print pairwise comparisons for most significant differences
        pairwise_results = statistical_tests['pairwise_comparisons']
        print(f"\n🔍 Significant Pairwise Comparisons (p < 0.05):")
        print("-" * 80)
        
        significant_comparisons = []
        
        for criteria in self.score_criteria:
            if criteria in pairwise_results:
                for comparison, result in pairwise_results[criteria].items():
                    if result['significant_at_0.05']:
                        significant_comparisons.append({
                            'criteria': criteria,
                            'comparison': comparison.replace('_vs_', ' vs '),
                            'p_value': result['mann_whitney_p_value'],
                            'mean_diff': result['mean_diff'],
                            'cohens_d': result['effect_size_cohens_d']
                        })
        
        # Sort by p-value
        significant_comparisons.sort(key=lambda x: x['p_value'])
        
        if significant_comparisons:
            print("Criteria            | Comparison           | p-value  | Mean Diff | Effect Size")
            print("-" * 80)
            for comp in significant_comparisons[:20]:  # Show top 20 most significant
                effect_size_desc = "Large" if abs(comp['cohens_d']) > 0.8 else ("Medium" if abs(comp['cohens_d']) > 0.5 else "Small")
                print(f"{comp['criteria']:<18} | {comp['comparison']:<19} | "
                      f"{comp['p_value']:8.6f} | {comp['mean_diff']:9.3f} | "
                      f"{comp['cohens_d']:4.2f} ({effect_size_desc})")
        else:
            print("No significant pairwise differences found at p < 0.05 level.")
        
        print(f"\n📝 Interpretation:")
        print(f"   *** p < 0.01 (highly significant)")
        print(f"   **  p < 0.05 (significant)")
        print(f"   Effect Size: Small (0.2-0.5), Medium (0.5-0.8), Large (>0.8)")
    
    def create_summary_dataframe(self, analysis_results: Dict) -> pd.DataFrame:
        """Create a pandas DataFrame with all the statistics for easy analysis."""
        box_plot_stats = analysis_results['box_plot_statistics']
        rows = []
        
        for agent in self.agents:
            if agent in box_plot_stats:
                for criteria in self.score_criteria:
                    if criteria in box_plot_stats[agent]:
                        stats = box_plot_stats[agent][criteria]
                        row = {
                            'agent': agent,
                            'criteria': criteria,
                            'min': stats['min'],
                            'q25': stats['q25'],
                            'median': stats['median'],
                            'q75': stats['q75'],
                            'max': stats['max'],
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'count': stats['count']
                        }
                        rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_box_plot_data(self, analysis_results: Dict, filename: str = "box_plot_statistics_with_pvalues.json"):
        """Save box plot statistics and p-values to JSON file."""
        with open(filename, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\n📁 Box plot statistics and p-values saved to: {filename}")
    
    def create_visualization(self, combined_scores: Dict):
        """Create box plots for all criteria and agents."""
        # Prepare data for plotting
        plot_data = []
        
        for agent in self.agents:
            if agent in combined_scores:
                for criteria in self.score_criteria:
                    scores = combined_scores[agent][criteria]
                    for score in scores:
                        plot_data.append({
                            'Agent': agent,
                            'Criteria': criteria.replace('_', ' ').title(),
                            'Score': score
                        })
        
        if not plot_data:
            print("Warning: No data available for visualization")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create box plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, criteria in enumerate(self.score_criteria):
            criteria_title = criteria.replace('_', ' ').title()
            criteria_data = df[df['Criteria'] == criteria_title]
            
            if not criteria_data.empty:
                sns.boxplot(data=criteria_data, x='Agent', y='Score', ax=axes[i])
                axes[i].set_title(f'{criteria_title}\n(Combined across all 4 models)', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Agent')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim(0.5, 5.5)
        
        plt.tight_layout()
        plt.savefig('pr_evaluation_box_plots.png', dpi=300, bbox_inches='tight')
        print("📊 Box plots saved to: pr_evaluation_box_plots.png")
        
        # Create overall comparison
        plt.figure(figsize=(15, 8))
        
        # Calculate overall scores for each agent
        overall_scores = []
        for agent in self.agents:
            if agent in combined_scores:
                all_agent_scores = []
                for criteria in self.score_criteria:
                    all_agent_scores.extend(combined_scores[agent][criteria])
                
                if all_agent_scores:
                    for score in all_agent_scores:
                        overall_scores.append({
                            'Agent': agent,
                            'Overall_Score': score
                        })
        
        if overall_scores:
            overall_df = pd.DataFrame(overall_scores)
            sns.boxplot(data=overall_df, x='Agent', y='Overall_Score')
            plt.title('Overall PR Description Quality Scores\n(All Criteria Combined, All Models)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Agent')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.ylim(0.5, 5.5)
            
            plt.tight_layout()
            plt.savefig('pr_evaluation_overall_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 Overall comparison plot saved to: pr_evaluation_overall_comparison.png")
    
    def run_analysis(self):
        """Run the complete analysis including statistical significance testing."""
        print("🚀 Starting PR Evaluation Results Analysis with Statistical Testing...")
        print(f"📁 Analyzing results from: {self.results_dir}")
        print(f"🤖 Models: {', '.join(self.models)}")
        print(f"👥 Agents: {', '.join(self.agents)}")
        print(f"📊 Criteria: {', '.join(self.score_criteria)}")
        print()
        
        # Load all results
        print("📖 Loading results from all models...")
        all_results = self.load_all_results()
        
        # Combine scores by agent
        print("\n🔗 Combining scores across all models for each agent...")
        combined_scores = self.combine_scores_by_agent(all_results)
        
        # Generate box plot statistics and p-values
        print("\n📊 Calculating box plot statistics and statistical significance...")
        analysis_results = self.generate_box_plot_summary(combined_scores)
        
        # Print summary
        self.print_box_plot_summary(analysis_results)
        
        # Save statistics
        self.save_box_plot_data(analysis_results)
        
        # Create DataFrame for further analysis
        df = self.create_summary_dataframe(analysis_results)
        df.to_csv("box_plot_statistics_with_pvalues.csv", index=False)
        print("📊 Box plot statistics saved to: box_plot_statistics_with_pvalues.csv")
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        try:
            self.create_visualization(combined_scores)
        except ImportError:
            print("⚠️  Matplotlib/Seaborn not available for plotting. Statistics saved to files.")
        except Exception as e:
            print(f"⚠️  Error creating visualizations: {e}")
        
        # Print final summary
        print(f"\n✅ Analysis complete!")
        print("📄 Generated files:")
        print("   - box_plot_statistics_with_pvalues.json")
        print("   - box_plot_statistics_with_pvalues.csv")
        print("   - pr_evaluation_box_plots.png (if matplotlib available)")
        print("   - pr_evaluation_overall_comparison.png (if matplotlib available)")
        
        return analysis_results, combined_scores

def main():
    """Main execution function."""
    analyzer = ResultsAnalyzer()
    analysis_results, combined_scores = analyzer.run_analysis()
    
    # Extract box plot stats from the new structure
    box_plot_stats = analysis_results['box_plot_statistics']
    
    # Optional: Print some interesting insights
    print(f"\n🔍 INSIGHTS:")
    
    # Find best performing agent overall
    agent_averages = {}
    for agent in analyzer.agents:
        if agent in box_plot_stats:
            all_means = [box_plot_stats[agent][criteria]['mean'] 
                        for criteria in analyzer.score_criteria 
                        if criteria in box_plot_stats[agent]]
            if all_means:
                agent_averages[agent] = sum(all_means) / len(all_means)
    
    if agent_averages:
        best_agent = max(agent_averages, key=agent_averages.get)
        worst_agent = min(agent_averages, key=agent_averages.get)
        
        print(f"🏆 Best performing agent: {best_agent} (avg: {agent_averages[best_agent]:.2f})")
        print(f"📉 Lowest performing agent: {worst_agent} (avg: {agent_averages[worst_agent]:.2f})")
        
        # Find best criteria for each agent
        print(f"\n🎯 Best criteria for each agent:")
        for agent in analyzer.agents:
            if agent in box_plot_stats:
                criteria_means = {criteria: box_plot_stats[agent][criteria]['mean'] 
                                for criteria in analyzer.score_criteria 
                                if criteria in box_plot_stats[agent]}
                if criteria_means:
                    best_criteria = max(criteria_means, key=criteria_means.get)
                    print(f"   {agent}: {best_criteria} ({criteria_means[best_criteria]:.2f})")

if __name__ == "__main__":
    main()