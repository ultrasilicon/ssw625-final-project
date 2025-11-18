#!/usr/bin/env python3
"""
Script to inspect sample PR data to show what information is available for evaluation.
"""

import sqlite3
from pathlib import Path

def inspect_sample_pr_data():
    """Show sample PR data structure."""
    print("Sample PR Data Inspection")
    print("=" * 50)
    
    if not Path("agent_pr.db").exists():
        print("❌ Database agent_pr.db not found!")
        return
    
    conn = sqlite3.connect("agent_pr.db")
    cursor = conn.cursor()
    
    # Get one sample PR from Copilot
    cursor.execute("""
        SELECT pr_id, title, body, agent 
        FROM Copilot_pull_requests 
        LIMIT 1
    """)
    pr_row = cursor.fetchone()
    
    if not pr_row:
        print("❌ No Copilot PRs found!")
        return
    
    pr_id, title, body, agent = pr_row
    
    print(f"Sample PR Data (Agent: {agent})")
    print("-" * 40)
    print(f"PR ID: {pr_id}")
    print(f"Title: {title}")
    print(f"\nDescription (first 500 chars):")
    print(f"{body[:500]}...")
    
    # Get commit details for this PR
    cursor.execute("""
        SELECT DISTINCT sha, message
        FROM Copilot_commit_details 
        WHERE pr_id = ?
        LIMIT 3
    """, (pr_id,))
    
    commits = cursor.fetchall()
    
    print(f"\nCommit Messages ({len(commits)} shown):")
    for i, (sha, message) in enumerate(commits, 1):
        print(f"{i}. {sha[:8]}: {message[:100]}...")
    
    # Get file changes
    cursor.execute("""
        SELECT filename, status, additions, deletions, patch
        FROM Copilot_commit_details 
        WHERE pr_id = ?
        LIMIT 5
    """, (pr_id,))
    
    files = cursor.fetchall()
    
    print(f"\nFile Changes ({len(files)} shown):")
    for i, (filename, status, additions, deletions, patch) in enumerate(files, 1):
        print(f"{i}. {filename} ({status}, +{additions}/-{deletions})")
        if patch:
            print(f"   Patch preview: {patch[:150]}...")
        print()
    
    conn.close()
    
    print("\nThis data is formatted and sent to GPT-4o-mini for evaluation based on the 6 criteria.")
    print("Each criterion gets its own API call with relevant data fields.")

if __name__ == "__main__":
    inspect_sample_pr_data()