#!/usr/bin/env python3
"""
Simple script to create box plots from the analysis results
This script shows how to use the prepared box plot data
"""

import csv
from pathlib import Path
import json

def create_simple_boxplot_summary():
    """Create a simple summary of box plot data without matplotlib."""
    
    results_dir = Path("analysis_results")
    criteria = [
        "coverage_purpose", "coverage_changes", "rationale_clarity",
        "commit_coverage", "testing_guidance", "readability"
    ]
    
    print("="*80)
    print("BOX PLOT DATA SUMMARY")
    print("="*80)
    
    for criterion in criteria:
        csv_file = results_dir / f"boxplot_{criterion}.csv"
        
        if not csv_file.exists():
            print(f"❌ File not found: {csv_file}")
            continue
        
        print(f"\n📊 {criterion.replace('_', ' ').title()}:")
        print("-" * 40)
        
        # Read and analyze the data
        agent_scores = {}
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                agent = row['Agent']
                score = int(row['Score'])
                
                if agent not in agent_scores:
                    agent_scores[agent] = []
                agent_scores[agent].append(score)
        
        # Calculate basic stats for each agent
        for agent, scores in agent_scores.items():
            mean_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            # Calculate quartiles (simple approximation)
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            median_idx = n // 2
            
            q1 = sorted_scores[q1_idx] if q1_idx < n else min_score
            median = sorted_scores[median_idx] if median_idx < n else mean_score
            q3 = sorted_scores[q3_idx] if q3_idx < n else max_score
            
            print(f"  {agent:<15}: Mean={mean_score:.2f}, "
                  f"Q1={q1}, Median={median}, Q3={q3}, "
                  f"Range=[{min_score}-{max_score}], n={len(scores)}")
    
    print("\n" + "="*80)
    print("📈 VISUALIZATION INSTRUCTIONS")
    print("="*80)
    
    print("\nTo create box plots with matplotlib/seaborn:")
    print("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plot style
plt.style.use('default')
sns.set_palette("Set2")

# Create subplots for all criteria
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('PR Description Quality Evaluation by Agent and Criteria', fontsize=16)

criteria = [
    'coverage_purpose', 'coverage_changes', 'rationale_clarity',
    'commit_coverage', 'testing_guidance', 'readability'
]

for i, criterion in enumerate(criteria):
    row = i // 3
    col = i % 3
    
    # Load data
    data = pd.read_csv(f'analysis_results/boxplot_{criterion}.csv')
    
    # Create box plot
    sns.boxplot(data=data, x='Agent', y='Score', ax=axes[row, col])
    axes[row, col].set_title(criterion.replace('_', ' ').title())
    axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), rotation=45)
    axes[row, col].set_ylim(0.5, 5.5)

plt.tight_layout()
plt.show()
""")
    
    print("\nTo create individual box plots:")
    print("""
# Example: Coverage Purpose
data = pd.read_csv('analysis_results/boxplot_coverage_purpose.csv')
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Agent', y='Score')
plt.title('Coverage Purpose Scores by Agent')
plt.xticks(rotation=45)
plt.ylim(0.5, 5.5)
plt.ylabel('Score (1-5)')
plt.tight_layout()
plt.show()
""")
    
    print("\n🗂️  Available Files:")
    for criterion in criteria:
        csv_file = results_dir / f"boxplot_{criterion}.csv"
        if csv_file.exists():
            print(f"  ✅ {csv_file}")
        else:
            print(f"  ❌ {csv_file}")

def print_key_insights():
    """Print key insights from the analysis."""
    
    results_file = Path("analysis_results") / "complete_analysis.json"
    
    if not results_file.exists():
        print("❌ Analysis results file not found.")
        return
    
    with open(results_file, 'r') as f:
        analysis = json.load(f)
    
    print("\n" + "="*80)
    print("🔍 KEY INSIGHTS FOR BOX PLOT INTERPRETATION")
    print("="*80)
    
    # Agent performance insights
    agent_summary = analysis['agent_summary']
    print("\n📊 Agent Performance Insights:")
    
    rankings = []
    for agent, stats in agent_summary.items():
        rankings.append((agent, stats['overall_mean']))
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top Performers:")
    for i, (agent, score) in enumerate(rankings[:3], 1):
        print(f"    {i}. {agent}: {score:.3f} average")
    
    print(f"\n  Areas to focus on:")
    criteria_analysis = analysis['criteria_analysis']
    worst_criteria = []
    for criterion, data in criteria_analysis.items():
        worst_criteria.append((criterion, data['overall']['mean']))
    worst_criteria.sort(key=lambda x: x[1])
    
    for criterion, score in worst_criteria[:3]:
        print(f"    • {criterion.replace('_', ' ').title()}: {score:.3f} (lowest average)")
    
    print(f"\n  Box Plot Interpretation Tips:")
    print(f"    • Box shows Q1, median, and Q3 (25th, 50th, 75th percentiles)")
    print(f"    • Whiskers show data range (excluding outliers)")
    print(f"    • Dots outside whiskers are outliers")
    print(f"    • Wider boxes = more variability in scores")
    print(f"    • Higher median line = better performance")
    
    print(f"\n  What to look for:")
    print(f"    • Consistent performers (narrow boxes, high medians)")
    print(f"    • Inconsistent performers (wide boxes, many outliers)")
    print(f"    • Criteria where all agents struggle (low overall distributions)")
    print(f"    • Agents with unique strengths/weaknesses")

def main():
    """Main function to run the summary."""
    create_simple_boxplot_summary()
    print_key_insights()

if __name__ == "__main__":
    main()