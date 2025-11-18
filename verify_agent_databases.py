#!/usr/bin/env python3
"""
Verification script to check the integrity of split agent databases.
"""

import sqlite3
import os
from pathlib import Path

def verify_agent_databases():
    """Verify that all agent databases contain the correct data."""
    print("="*80)
    print("VERIFICATION OF AGENT DATABASES")
    print("="*80)
    
    agent_db_dir = "agent_databases"
    
    if not Path(agent_db_dir).exists():
        print(f"Directory {agent_db_dir} not found!")
        return
    
    # Get all .db files in the agent_databases directory
    db_files = list(Path(agent_db_dir).glob("*.db"))
    
    total_prs_split = 0
    total_commits_split = 0
    
    for db_file in sorted(db_files):
        agent_name = db_file.stem.replace('_', ' ')
        print(f"\nAgent: {agent_name}")
        print("-" * 40)
        
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Count PRs
            cursor.execute("SELECT COUNT(*) FROM pull_requests")
            pr_count = cursor.fetchone()[0]
            
            # Count commit details
            cursor.execute("SELECT COUNT(*) FROM commit_details")
            commit_count = cursor.fetchone()[0]
            
            # Get agent from first PR (should be consistent)
            cursor.execute("SELECT agent FROM pull_requests LIMIT 1")
            db_agent = cursor.fetchone()
            
            # Get file size
            file_size = db_file.stat().st_size / (1024 * 1024)  # MB
            
            print(f"  Database: {db_file.name}")
            print(f"  File size: {file_size:.1f} MB")
            print(f"  Pull requests: {pr_count:,}")
            print(f"  Commit details: {commit_count:,}")
            print(f"  Agent in DB: {db_agent[0] if db_agent else 'None'}")
            
            total_prs_split += pr_count
            total_commits_split += commit_count
            
            conn.close()
            
        except Exception as e:
            print(f"  Error reading {db_file}: {e}")
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total PRs in split databases: {total_prs_split:,}")
    print(f"Total commit details in split databases: {total_commits_split:,}")
    
    # Compare with original database
    try:
        original_conn = sqlite3.connect("research_pr_analysis.db")
        original_cursor = original_conn.cursor()
        
        original_cursor.execute("SELECT COUNT(*) FROM pull_requests")
        original_prs = original_cursor.fetchone()[0]
        
        original_cursor.execute("SELECT COUNT(*) FROM commit_details")
        original_commits = original_cursor.fetchone()[0]
        
        print(f"Original database PRs: {original_prs:,}")
        print(f"Original database commit details: {original_commits:,}")
        
        print(f"\nData integrity check:")
        if total_prs_split == original_prs:
            print("✅ PR counts match!")
        else:
            print(f"❌ PR count mismatch: {total_prs_split} vs {original_prs}")
            
        if total_commits_split == original_commits:
            print("✅ Commit detail counts match!")
        else:
            print(f"❌ Commit detail count mismatch: {total_commits_split} vs {original_commits}")
        
        original_conn.close()
        
    except Exception as e:
        print(f"Error reading original database: {e}")

if __name__ == "__main__":
    verify_agent_databases()