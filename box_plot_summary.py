#!/usr/bin/env python3
"""
Simple script to extract and display box plot statistics for PR evaluation results
Focus on: Min, Max, 25th percentile, 75th percentile, and Median for each score and agent
"""

import json
import pandas as pd
from pathlib import Path

def print_box_plot_data():
    """Print box plot data in a clean format for each agent and score."""
    
    # Load the generated statistics
    if Path("box_plot_statistics.json").exists():
        with open("box_plot_statistics.json", 'r') as f:
            stats = json.load(f)
    else:
        print("Error: box_plot_statistics.json not found. Please run analyze_results.py first.")
        return
    
    agents = ["Claude_Code", "Copilot", "Cursor", "Devin", "OpenAI_Codex"]
    criteria = ["coverage_purpose", "coverage_changes", "rationale_clarity", 
                "commit_coverage", "testing_guidance", "readability"]
    
    print("=" * 90)
    print("BOX PLOT DATA FOR PR EVALUATION RESULTS")
    print("Combined data from all 4 evaluation models (DeepSeek-V3.1, Deepseek-V3-0324, GPT-4.1, Grok-3)")
    print("=" * 90)
    print("Format: Agent | Min | Q25 | Median | Q75 | Max | (1200 evaluations per agent)")
    print("=" * 90)
    
    for criteria_name in criteria:
        print(f"\n📊 {criteria_name.upper().replace('_', ' ')}")
        print("-" * 70)
        
        for agent in agents:
            if agent in stats and criteria_name in stats[agent]:
                data = stats[agent][criteria_name]
                print(f"{agent:<15} | {data['min']:3.1f} | {data['q25']:3.1f} | "
                      f"{data['median']:3.1f} | {data['q75']:3.1f} | {data['max']:3.1f}")
    
    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)
    
    # Calculate overall performance for each agent
    print("\n🏆 OVERALL AGENT PERFORMANCE (Average across all criteria):")
    print("-" * 50)
    
    agent_averages = {}
    for agent in agents:
        if agent in stats:
            all_medians = [stats[agent][criteria_name]['median'] 
                          for criteria_name in criteria 
                          if criteria_name in stats[agent]]
            if all_medians:
                avg_median = sum(all_medians) / len(all_medians)
                agent_averages[agent] = avg_median
                print(f"{agent:<15}: {avg_median:.2f}")
    
    # Rank agents
    if agent_averages:
        sorted_agents = sorted(agent_averages.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🥇 Agent Rankings (by median score):")
        for i, (agent, score) in enumerate(sorted_agents, 1):
            print(f"  {i}. {agent}: {score:.2f}")
    
    print("\n📊 BEST PERFORMING CRITERIA BY AGENT:")
    print("-" * 50)
    
    for agent in agents:
        if agent in stats:
            criteria_medians = {criteria_name: stats[agent][criteria_name]['median'] 
                               for criteria_name in criteria 
                               if criteria_name in stats[agent]}
            if criteria_medians:
                best_criteria = max(criteria_medians, key=criteria_medians.get)
                worst_criteria = min(criteria_medians, key=criteria_medians.get)
                print(f"{agent:<15}: Best = {best_criteria} ({criteria_medians[best_criteria]:.1f}), "
                      f"Worst = {worst_criteria} ({criteria_medians[worst_criteria]:.1f})")

    print("\n📈 CRITERIA PERFORMANCE ACROSS ALL AGENTS:")
    print("-" * 50)
    
    criteria_averages = {}
    for criteria_name in criteria:
        all_medians = [stats[agent][criteria_name]['median'] 
                      for agent in agents 
                      if agent in stats and criteria_name in stats[agent]]
        if all_medians:
            avg_median = sum(all_medians) / len(all_medians)
            criteria_averages[criteria_name] = avg_median
    
    if criteria_averages:
        sorted_criteria = sorted(criteria_averages.items(), key=lambda x: x[1], reverse=True)
        for i, (criteria_name, score) in enumerate(sorted_criteria, 1):
            print(f"  {i}. {criteria_name.replace('_', ' ').title()}: {score:.2f}")

def export_for_plotting():
    """Export data in format suitable for box plotting."""
    
    if not Path("box_plot_statistics.csv").exists():
        print("Error: box_plot_statistics.csv not found. Please run analyze_results.py first.")
        return
    
    df = pd.read_csv("box_plot_statistics.csv")
    
    print(f"\n📁 Box plot data available in:")
    print(f"   - box_plot_statistics.csv (detailed format)")
    print(f"   - box_plot_statistics.json (nested format)")
    print(f"   - pr_evaluation_box_plots.png (visualization)")
    print(f"   - pr_evaluation_overall_comparison.png (overall comparison)")
    
    print(f"\n📊 Dataset summary:")
    print(f"   - Total data points: {len(df)}")
    print(f"   - Agents: {df['agent'].nunique()}")
    print(f"   - Criteria: {df['criteria'].nunique()}")
    print(f"   - Evaluations per agent-criteria: {df['count'].iloc[0]:,}")
    
    # Show sample data format
    print(f"\n📋 Sample data format (first 3 rows):")
    print(df[['agent', 'criteria', 'min', 'q25', 'median', 'q75', 'max']].head(3).to_string(index=False))

def main():
    """Main function."""
    print_box_plot_data()
    export_for_plotting()
    
    print(f"\n✅ Box plot analysis complete!")
    print(f"💡 Use this data to create box plots showing distribution of scores for each agent and criteria.")
    print(f"🎯 Key insight: Each agent has 1,200 evaluations (300 PRs × 4 models) per criteria.")

if __name__ == "__main__":
    main()