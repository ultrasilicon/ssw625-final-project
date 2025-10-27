#!/usr/bin/env python3
"""
Optimized script to check if all PRs in human_pull_request.csv and AI_pull_request.csv
exist in pr_commit_details.csv. Uses efficient pandas operations.
"""

import pandas as pd
import time
from pathlib import Path

def check_pr_coverage_optimized(data_dir: str = "data"):
    """Optimized function to check PR coverage using pandas operations."""
    start_time = time.time()
    
    # File paths
    human_pr_file = f"{data_dir}/human_pull_request.csv"
    ai_pr_file = f"{data_dir}/AI_pull_request.csv"
    commit_details_file = f"{data_dir}/pr_commit_details.csv"
    
    print("=== Checking PR Coverage (Optimized) ===\n")
    
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
    
    # Process commit details file efficiently
    print(f"\nProcessing {Path(commit_details_file).name}...")
    print("This may take a few minutes due to file size (845MB)...")
    
    # Read only the pr_id column in chunks and find unique values
    chunk_size = 100000
    found_pr_ids = set()
    processed_rows = 0
    
    print("Reading commit details in chunks...")
    for chunk_num, chunk in enumerate(pd.read_csv(commit_details_file, 
                                                 usecols=['pr_id'], 
                                                 chunksize=chunk_size), 1):
        # Get unique PR IDs from this chunk that match our targets
        chunk_pr_ids = set(chunk['pr_id'].unique())
        chunk_matches = chunk_pr_ids.intersection(all_target_pr_ids)
        found_pr_ids.update(chunk_matches)
        
        processed_rows += len(chunk)
        
        # Progress update every 20 chunks (2M rows)
        if chunk_num % 20 == 0:
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
    print(f"\nTotal PR IDs found in commit details: {len(found_pr_ids):,}")
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
    
    # Check if there's any overlap in missing PRs
    missing_overlap = human_missing.intersection(ai_missing)
    if missing_overlap:
        print(f"\nPR IDs missing from BOTH datasets: {len(missing_overlap):,}")
        print(f"Sample: {sorted(list(missing_overlap))[:5]}")
    
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
        'processed_rows': processed_rows
    }

if __name__ == "__main__":
    # Change to the directory containing the script
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        results = check_pr_coverage_optimized()
        
        # Save detailed results to a summary file
        with open("pr_coverage_summary.txt", "w") as f:
            f.write("PR Coverage Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {results['processing_time']:.2f} seconds\n")
            f.write(f"Rows Processed: {results['processed_rows']:,}\n\n")
            
            f.write("COVERAGE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total PR IDs to check: {results['total_target_prs']:,}\n")
            f.write(f"Total found in commit details: {results['total_found']:,}\n")
            f.write(f"Overall coverage: {results['coverage_percentage']:.2f}%\n\n")
            
            f.write("DETAILED BREAKDOWN\n")
            f.write("-" * 20 + "\n")
            f.write(f"Human PRs: {results['human_found']:,}/{results['human_total']:,} found ({results['human_coverage']:.2f}%)\n")
            f.write(f"AI PRs: {results['ai_found']:,}/{results['ai_total']:,} found ({results['ai_coverage']:.2f}%)\n")
            f.write(f"Human PRs missing: {results['human_missing']:,}\n")
            f.write(f"AI PRs missing: {results['ai_missing']:,}\n")
        
        print("\nDetailed summary saved to pr_coverage_summary.txt")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise