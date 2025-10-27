#!/usr/bin/env python3
"""
Script to check if all PRs in human_pull_request.csv and AI_pull_request.csv
exist in pr_commits.csv. This checks a different commits dataset.
"""

import pandas as pd
import time
from pathlib import Path

def check_pr_commits_coverage(data_dir: str = "data"):
    """Check PR coverage in pr_commits.csv file."""
    start_time = time.time()
    
    # File paths
    human_pr_file = f"{data_dir}/human_pull_request.csv"
    ai_pr_file = f"{data_dir}/AI_pull_request.csv"
    pr_commits_file = f"{data_dir}/pr_commits.csv"
    
    print("=== Checking PR Coverage in pr_commits.csv ===\n")
    
    # Load PR IDs from human and AI PR files
    print("Loading Human PR IDs...")
    human_df = pd.read_csv(human_pr_file, usecols=['id'])
    human_pr_ids = set(human_df['id'].tolist())
    print(f"  Loaded {len(human_pr_ids):,} unique Human PR IDs")
    
    print("Loading AI PR IDs...")
    ai_df = pd.read_csv(ai_pr_file, usecols=['id'])
    ai_pr_ids = set(ai_df['id'].tolist())
    print(f"  Loaded {len(ai_pr_ids):,} unique AI PR IDs")
    
    # Combine all target PR IDs
    all_target_pr_ids = human_pr_ids.union(ai_pr_ids)
    overlap = human_pr_ids.intersection(ai_pr_ids)
    
    print(f"\nTotal unique PR IDs to check: {len(all_target_pr_ids):,}")
    print(f"  Human PRs: {len(human_pr_ids):,}")
    print(f"  AI PRs: {len(ai_pr_ids):,}")
    print(f"  Overlap: {len(overlap):,}")
    
    # Check file size
    file_size_mb = Path(pr_commits_file).stat().st_size / (1024 * 1024)
    print(f"\nProcessing {Path(pr_commits_file).name} ({file_size_mb:.1f} MB)...")
    
    # Process pr_commits file efficiently
    chunk_size = 50000
    found_pr_ids = set()
    processed_rows = 0
    
    print("Reading pr_commits in chunks...")
    for chunk_num, chunk in enumerate(pd.read_csv(pr_commits_file, 
                                                 usecols=['pr_id'], 
                                                 chunksize=chunk_size), 1):
        # Get unique PR IDs from this chunk that match our targets
        chunk_pr_ids = set(chunk['pr_id'].unique())
        chunk_matches = chunk_pr_ids.intersection(all_target_pr_ids)
        found_pr_ids.update(chunk_matches)
        
        processed_rows += len(chunk)
        
        # Progress update every 10 chunks (500k rows)
        if chunk_num % 10 == 0:
            print(f"  Processed {processed_rows:,} rows, found {len(found_pr_ids):,} matching PRs so far...")
        
        # Early exit if we've found all target PRs
        if len(found_pr_ids) == len(all_target_pr_ids):
            print(f"  Found all target PRs! Stopping early at {processed_rows:,} rows.")
            break
    
    print(f"Finished processing {processed_rows:,} rows")
    
    # Calculate coverage
    human_found = found_pr_ids.intersection(human_pr_ids)
    ai_found = found_pr_ids.intersection(ai_pr_ids)
    human_missing = human_pr_ids - found_pr_ids
    ai_missing = ai_pr_ids - found_pr_ids
    
    # Results
    processing_time = time.time() - start_time
    print("\n=== RESULTS ===")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"\nTotal PR IDs found in pr_commits: {len(found_pr_ids):,}")
    print(f"Overall coverage: {len(found_pr_ids) / len(all_target_pr_ids) * 100:.2f}%")
    
    print(f"\nHuman PRs:")
    print(f"  Total: {len(human_pr_ids):,}")
    print(f"  Found: {len(human_found):,} ({len(human_found) / len(human_pr_ids) * 100:.2f}%)")
    print(f"  Missing: {len(human_missing):,}")
    
    print(f"\nAI PRs:")
    print(f"  Total: {len(ai_pr_ids):,}")
    print(f"  Found: {len(ai_found):,} ({len(ai_found) / len(ai_pr_ids) * 100:.2f}%)")
    print(f"  Missing: {len(ai_missing):,}")
    
    # Show some examples of missing PRs
    if human_missing:
        missing_sample = sorted(list(human_missing))[:10]
        print(f"\nFirst 10 missing Human PR IDs: {missing_sample}")
    
    if ai_missing:
        missing_sample = sorted(list(ai_missing))[:10]
        print(f"First 10 missing AI PR IDs: {missing_sample}")
    
    # Show some examples of found PRs for each type
    if human_found:
        found_sample = sorted(list(human_found))[:5]
        print(f"\nFirst 5 found Human PR IDs: {found_sample}")
    
    if ai_found:
        found_sample = sorted(list(ai_found))[:5]
        print(f"First 5 found AI PR IDs: {found_sample}")
    
    # Check if there's any overlap in missing PRs
    missing_overlap = human_missing.intersection(ai_missing)
    if missing_overlap:
        print(f"\nPR IDs missing from BOTH datasets: {len(missing_overlap):,}")
        print(f"Sample: {sorted(list(missing_overlap))[:5]}")
    
    # Additional analysis: get unique PR count in pr_commits
    print(f"\nAnalyzing pr_commits.csv structure...")
    all_pr_commits_ids = set()
    for chunk in pd.read_csv(pr_commits_file, usecols=['pr_id'], chunksize=chunk_size):
        all_pr_commits_ids.update(chunk['pr_id'].unique())
    
    print(f"Total unique PR IDs in pr_commits.csv: {len(all_pr_commits_ids):,}")
    
    # Find PRs in pr_commits that are not in our target datasets
    extra_prs = all_pr_commits_ids - all_target_pr_ids
    print(f"PR IDs in pr_commits but not in human/AI datasets: {len(extra_prs):,}")
    if extra_prs:
        print(f"Sample extra PR IDs: {sorted(list(extra_prs))[:5]}")
    
    return {
        'total_target_prs': len(all_target_pr_ids),
        'total_found': len(found_pr_ids),
        'coverage_percentage': len(found_pr_ids) / len(all_target_pr_ids) * 100,
        'human_total': len(human_pr_ids),
        'human_found': len(human_found),
        'human_missing': len(human_missing),
        'human_coverage': len(human_found) / len(human_pr_ids) * 100,
        'ai_total': len(ai_pr_ids),
        'ai_found': len(ai_found),
        'ai_missing': len(ai_missing),
        'ai_coverage': len(ai_found) / len(ai_pr_ids) * 100,
        'processing_time': processing_time,
        'processed_rows': processed_rows,
        'total_pr_commits_ids': len(all_pr_commits_ids),
        'extra_prs_count': len(extra_prs)
    }

if __name__ == "__main__":
    # Change to the directory containing the script
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        results = check_pr_commits_coverage()
        
        # Save detailed results to a summary file
        with open("pr_commits_coverage_summary.txt", "w") as f:
            f.write("PR Coverage Analysis - pr_commits.csv\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {results['processing_time']:.2f} seconds\n")
            f.write(f"Rows Processed: {results['processed_rows']:,}\n\n")
            
            f.write("COVERAGE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total PR IDs to check: {results['total_target_prs']:,}\n")
            f.write(f"Total found in pr_commits: {results['total_found']:,}\n")
            f.write(f"Overall coverage: {results['coverage_percentage']:.2f}%\n\n")
            
            f.write("DETAILED BREAKDOWN\n")
            f.write("-" * 20 + "\n")
            f.write(f"Human PRs: {results['human_found']:,}/{results['human_total']:,} found ({results['human_coverage']:.2f}%)\n")
            f.write(f"AI PRs: {results['ai_found']:,}/{results['ai_total']:,} found ({results['ai_coverage']:.2f}%)\n")
            f.write(f"Human PRs missing: {results['human_missing']:,}\n")
            f.write(f"AI PRs missing: {results['ai_missing']:,}\n\n")
            
            f.write("ADDITIONAL INSIGHTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total unique PR IDs in pr_commits.csv: {results['total_pr_commits_ids']:,}\n")
            f.write(f"PR IDs in pr_commits but not in human/AI datasets: {results['extra_prs_count']:,}\n")
        
        print("\nDetailed summary saved to pr_commits_coverage_summary.txt")
        
        # Quick comparison with previous results if available
        if Path("pr_coverage_summary.txt").exists():
            print("\n=== COMPARISON WITH pr_commit_details.csv ===")
            print("Previous analysis (pr_commit_details.csv):")
            print("  - Human PRs: 0/6,618 found (0.00%)")
            print("  - AI PRs: 33,580/33,596 found (99.95%)")
            print(f"Current analysis (pr_commits.csv):")
            print(f"  - Human PRs: {results['human_found']:,}/{results['human_total']:,} found ({results['human_coverage']:.2f}%)")
            print(f"  - AI PRs: {results['ai_found']:,}/{results['ai_total']:,} found ({results['ai_coverage']:.2f}%)")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise