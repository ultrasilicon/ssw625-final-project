#!/usr/bin/env python3
"""
Script to check if all PR IDs from human_pull_request.csv exist in pr_commits.parquet
"""

import pandas as pd

def check_pr_ids_coverage():
    print("Loading data files...")
    
    # Load human pull requests CSV
    print("  - Reading data/human_pull_request.csv...")
    human_pr_df = pd.read_csv('data/human_pull_request.csv')
    human_pr_ids = set(human_pr_df['id'].unique())
    print(f"    Found {len(human_pr_ids)} unique PR IDs in human_pull_request.csv")
    
    # Load PR commits parquet
    print("  - Reading data/pr_commits.parquet...")
    pr_commits_df = pd.read_parquet('data/pr_commits.parquet')
    pr_commits_ids = set(pr_commits_df['pr_id'].unique())
    print(f"    Found {len(pr_commits_ids)} unique PR IDs in pr_commits.parquet")
    
    print("\nAnalyzing coverage...")
    
    # Check which PR IDs from CSV are in parquet
    pr_ids_in_parquet = human_pr_ids.intersection(pr_commits_ids)
    pr_ids_missing = human_pr_ids - pr_commits_ids
    
    # Calculate coverage
    coverage_percentage = (len(pr_ids_in_parquet) / len(human_pr_ids) * 100) if len(human_pr_ids) > 0 else 0
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Total PR IDs in human_pull_request.csv: {len(human_pr_ids)}")
    print(f"PR IDs found in pr_commits.parquet:    {len(pr_ids_in_parquet)}")
    print(f"PR IDs missing from pr_commits.parquet: {len(pr_ids_missing)}")
    print(f"Coverage: {coverage_percentage:.2f}%")
    print(f"{'='*60}")
    
    if len(pr_ids_missing) == 0:
        print("\n✓ SUCCESS: All PR IDs from human_pull_request.csv are present in pr_commits.parquet!")
    else:
        print(f"\n✗ INCOMPLETE: {len(pr_ids_missing)} PR IDs are missing from pr_commits.parquet")
        
        # Show first 10 missing PR IDs as examples
        if len(pr_ids_missing) > 0:
            missing_list = sorted(list(pr_ids_missing))[:10]
            print(f"\nFirst {min(10, len(pr_ids_missing))} missing PR IDs:")
            for pr_id in missing_list:
                print(f"  - {pr_id}")
            
            if len(pr_ids_missing) > 10:
                print(f"  ... and {len(pr_ids_missing) - 10} more")
    
    # Also check for PR IDs in parquet that are not in CSV (for information)
    extra_pr_ids = pr_commits_ids - human_pr_ids
    if len(extra_pr_ids) > 0:
        print(f"\nNote: pr_commits.parquet contains {len(extra_pr_ids)} PR IDs that are NOT in human_pull_request.csv")
    
    return len(pr_ids_missing) == 0

if __name__ == "__main__":
    success = check_pr_ids_coverage()
    exit(0 if success else 1)

