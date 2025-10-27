#!/usr/bin/env python3
"""
Example script showing how to query a specific PR and its associated commits and file changes.
"""

from query_aidev_database import AIDevQueryHelper
import pandas as pd

def example_pr_analysis():
    """Example analysis of a specific PR."""
    helper = AIDevQueryHelper()
    
    try:
        # Get a PR with many file changes for demonstration
        top_pr_query = """
        SELECT id, number, title, commit_count, file_changes_count, repo_url
        FROM pull_requests 
        WHERE file_changes_count > 50
        ORDER BY file_changes_count DESC 
        LIMIT 1
        """
        
        top_pr = pd.read_sql_query(top_pr_query, helper.conn)
        
        if not top_pr.empty:
            pr_id = top_pr.iloc[0]['id']
            print(f"Analyzing PR #{top_pr.iloc[0]['number']}: {top_pr.iloc[0]['title']}")
            print(f"Repository: {top_pr.iloc[0]['repo_url'].split('/')[-1]}")
            print(f"File changes: {top_pr.iloc[0]['file_changes_count']:,}")
            print(f"Commits: {top_pr.iloc[0]['commit_count']:,}")
            print("-" * 80)
            
            # Get full PR details
            pr_details = helper.get_pr_with_commits(pr_id)
            
            if pr_details:
                print(f"\nCommits ({len(pr_details['commits'])} total):")
                for _, commit in pr_details['commits'].head(3).iterrows():
                    print(f"  {commit['sha'][:8]} by {commit['author']}: {commit['message'][:60]}...")
                
                if len(pr_details['commits']) > 3:
                    print(f"  ... and {len(pr_details['commits']) - 3} more commits")
                
                print(f"\nFile Changes (showing first 10 of {len(pr_details['file_changes'])}):")
                for _, file_change in pr_details['file_changes'].head(10).iterrows():
                    status_icon = {"added": "+", "modified": "~", "removed": "-"}.get(file_change['status'], "?")
                    print(f"  {status_icon} {file_change['filename']} (+{file_change['additions']}/-{file_change['deletions']})")
                
                # File type breakdown for this PR
                file_types = {}
                for _, file_change in pr_details['file_changes'].iterrows():
                    filename = file_change['filename']
                    if '.' in filename:
                        ext = filename.split('.')[-1].lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
                    else:
                        file_types['no_ext'] = file_types.get('no_ext', 0) + 1
                
                print(f"\nFile Type Breakdown:")
                for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  .{ext}: {count} files")
        
        print("\n" + "="*80)
        print("Database Summary:")
        print(f"Total PRs: {33596:,}")
        print(f"Total Commits: {86315:,}")  
        print(f"Total File Changes: {711923:,}")
        print(f"Database Size: 954.1 MB")
        print("\nTo explore more, use the query_aidev_database.py script!")
        
    finally:
        helper.close()

if __name__ == "__main__":
    example_pr_analysis()