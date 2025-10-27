#!/usr/bin/env python3
"""
Helper script to query and explore the AI development analysis SQLite database.
Provides common queries and analysis functions.
"""

import sqlite3
import pandas as pd
from pathlib import Path

class AIDevQueryHelper:
    def __init__(self, db_name: str = "aidev_analysis.db"):
        self.db_name = db_name
        if not Path(db_name).exists():
            raise FileNotFoundError(f"Database {db_name} not found. Run create_aidev_database.py first.")
        self.conn = sqlite3.connect(db_name)
    
    def get_pr_with_commits(self, pr_id: int):
        """Get a specific PR with all its commits and file changes."""
        query = """
        SELECT 
            pr.id, pr.number, pr.title, pr.agent, pr.state, pr.created_at,
            pr.commit_count, pr.file_changes_count
        FROM pull_requests pr
        WHERE pr.id = ?
        """
        pr_info = pd.read_sql_query(query, self.conn, params=[pr_id])
        
        if pr_info.empty:
            print(f"No PR found with ID: {pr_id}")
            return None
        
        # Get commits for this PR
        commits_query = """
        SELECT sha, author, committer, message, 
               commit_stats_total, commit_stats_additions, commit_stats_deletions
        FROM commits 
        WHERE pr_id = ?
        ORDER BY sha
        """
        commits = pd.read_sql_query(commits_query, self.conn, params=[pr_id])
        
        # Get file changes for this PR
        files_query = """
        SELECT commit_sha, filename, status, additions, deletions, changes
        FROM file_changes 
        WHERE pr_id = ?
        ORDER BY commit_sha, filename
        """
        file_changes = pd.read_sql_query(files_query, self.conn, params=[pr_id])
        
        return {
            'pr_info': pr_info,
            'commits': commits,
            'file_changes': file_changes
        }
    
    def get_top_repositories(self, limit: int = 10):
        """Get repositories with most PRs."""
        query = """
        SELECT 
            repo_url,
            COUNT(*) as pr_count,
            AVG(commit_count) as avg_commits_per_pr,
            AVG(file_changes_count) as avg_files_per_pr,
            SUM(file_changes_count) as total_file_changes
        FROM pull_requests 
        GROUP BY repo_url 
        ORDER BY pr_count DESC 
        LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=[limit])
    
    def get_file_extension_analysis(self):
        """Analyze file extensions and programming languages."""
        query = """
        SELECT 
            CASE 
                WHEN filename LIKE '%.py' THEN 'Python (.py)'
                WHEN filename LIKE '%.js' THEN 'JavaScript (.js)'
                WHEN filename LIKE '%.ts' THEN 'TypeScript (.ts)'
                WHEN filename LIKE '%.tsx' THEN 'TypeScript React (.tsx)'
                WHEN filename LIKE '%.jsx' THEN 'JavaScript React (.jsx)'
                WHEN filename LIKE '%.java' THEN 'Java (.java)'
                WHEN filename LIKE '%.cpp' OR filename LIKE '%.cc' OR filename LIKE '%.cxx' THEN 'C++ (.cpp/.cc/.cxx)'
                WHEN filename LIKE '%.c' THEN 'C (.c)'
                WHEN filename LIKE '%.h' OR filename LIKE '%.hpp' THEN 'Header (.h/.hpp)'
                WHEN filename LIKE '%.go' THEN 'Go (.go)'
                WHEN filename LIKE '%.rs' THEN 'Rust (.rs)'
                WHEN filename LIKE '%.php' THEN 'PHP (.php)'
                WHEN filename LIKE '%.rb' THEN 'Ruby (.rb)'
                WHEN filename LIKE '%.swift' THEN 'Swift (.swift)'
                WHEN filename LIKE '%.kt' THEN 'Kotlin (.kt)'
                WHEN filename LIKE '%.scala' THEN 'Scala (.scala)'
                WHEN filename LIKE '%.md' THEN 'Markdown (.md)'
                WHEN filename LIKE '%.json' THEN 'JSON (.json)'
                WHEN filename LIKE '%.yml' OR filename LIKE '%.yaml' THEN 'YAML (.yml/.yaml)'
                WHEN filename LIKE '%.xml' THEN 'XML (.xml)'
                WHEN filename LIKE '%.html' THEN 'HTML (.html)'
                WHEN filename LIKE '%.css' THEN 'CSS (.css)'
                WHEN filename LIKE '%.scss' OR filename LIKE '%.sass' THEN 'Sass/SCSS (.scss/.sass)'
                WHEN filename LIKE '%.sql' THEN 'SQL (.sql)'
                WHEN filename LIKE '%.sh' THEN 'Shell Script (.sh)'
                WHEN filename LIKE 'Dockerfile%' THEN 'Docker (Dockerfile)'
                WHEN filename LIKE '%.toml' THEN 'TOML (.toml)'
                WHEN filename LIKE '%.ini' OR filename LIKE '%.cfg' THEN 'Config (.ini/.cfg)'
                ELSE 'Other'
            END as file_type,
            COUNT(*) as change_count,
            SUM(additions) as total_additions,
            SUM(deletions) as total_deletions,
            COUNT(DISTINCT pr_id) as affected_prs
        FROM file_changes 
        GROUP BY file_type 
        ORDER BY change_count DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_pr_complexity_analysis(self):
        """Analyze PR complexity by commits and file changes."""
        query = """
        SELECT 
            CASE 
                WHEN commit_count = 1 THEN '1 commit'
                WHEN commit_count BETWEEN 2 AND 5 THEN '2-5 commits'
                WHEN commit_count BETWEEN 6 AND 10 THEN '6-10 commits'
                WHEN commit_count BETWEEN 11 AND 20 THEN '11-20 commits'
                ELSE '20+ commits'
            END as commit_range,
            CASE 
                WHEN file_changes_count BETWEEN 1 AND 5 THEN '1-5 files'
                WHEN file_changes_count BETWEEN 6 AND 15 THEN '6-15 files'
                WHEN file_changes_count BETWEEN 16 AND 50 THEN '16-50 files'
                WHEN file_changes_count BETWEEN 51 AND 100 THEN '51-100 files'
                ELSE '100+ files'
            END as file_range,
            COUNT(*) as pr_count,
            AVG(commit_count) as avg_commits,
            AVG(file_changes_count) as avg_file_changes
        FROM pull_requests 
        WHERE commit_count > 0
        GROUP BY commit_range, file_range
        ORDER BY pr_count DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def search_prs_by_title(self, search_term: str, limit: int = 20):
        """Search PRs by title containing specific terms."""
        query = """
        SELECT 
            id, number, title, agent, state, created_at,
            commit_count, file_changes_count, repo_url
        FROM pull_requests 
        WHERE title LIKE ? 
        ORDER BY created_at DESC
        LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=[f'%{search_term}%', limit])
    
    def get_author_analysis(self, limit: int = 20):
        """Analyze top commit authors."""
        query = """
        SELECT 
            author,
            COUNT(*) as commit_count,
            COUNT(DISTINCT pr_id) as pr_count,
            COUNT(DISTINCT pr_id) * 100.0 / (SELECT COUNT(*) FROM pull_requests) as pr_percentage
        FROM commits 
        GROUP BY author 
        ORDER BY commit_count DESC
        LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=[limit])
    
    def export_sample_data(self, output_dir: str = "exports"):
        """Export sample data to CSV files for analysis."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Sample PRs
        sample_prs = pd.read_sql_query("""
            SELECT * FROM pull_requests 
            ORDER BY file_changes_count DESC 
            LIMIT 1000
        """, self.conn)
        sample_prs.to_csv(f"{output_dir}/sample_prs.csv", index=False)
        
        # Sample commits
        sample_commits = pd.read_sql_query("""
            SELECT * FROM commits 
            WHERE pr_id IN (SELECT id FROM pull_requests ORDER BY file_changes_count DESC LIMIT 100)
        """, self.conn)
        sample_commits.to_csv(f"{output_dir}/sample_commits.csv", index=False)
        
        # File extension summary
        file_analysis = self.get_file_extension_analysis()
        file_analysis.to_csv(f"{output_dir}/file_type_analysis.csv", index=False)
        
        print(f"Sample data exported to {output_dir}/ directory")
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Demo function showing various queries."""
    helper = AIDevQueryHelper()
    
    try:
        print("AI Development Analysis - Database Query Examples")
        print("="*60)
        
        # Top repositories
        print("\n1. Top 10 Repositories by PR Count:")
        top_repos = helper.get_top_repositories(10)
        for _, row in top_repos.iterrows():
            repo_name = row['repo_url'].split('/')[-1] if row['repo_url'] else 'Unknown'
            print(f"   {repo_name}: {row['pr_count']:,} PRs, {row['avg_commits_per_pr']:.1f} avg commits, {row['avg_files_per_pr']:.1f} avg file changes")
        
        # File type analysis
        print("\n2. Top 10 File Types by Change Count:")
        file_analysis = helper.get_file_extension_analysis().head(10)
        for _, row in file_analysis.iterrows():
            print(f"   {row['file_type']}: {row['change_count']:,} changes in {row['affected_prs']:,} PRs")
        
        # Author analysis
        print("\n3. Top 10 Commit Authors:")
        authors = helper.get_author_analysis(10)
        for _, row in authors.iterrows():
            print(f"   {row['author']}: {row['commit_count']:,} commits in {row['pr_count']:,} PRs ({row['pr_percentage']:.1f}%)")
        
        # Search example
        print("\n4. Example: PRs with 'fix' in title (first 5):")
        fix_prs = helper.search_prs_by_title('fix', 5)
        for _, row in fix_prs.iterrows():
            print(f"   PR #{row['number']}: {row['title'][:80]}...")
        
        # Complexity analysis sample
        print("\n5. PR Complexity Distribution (top 5 categories):")
        complexity = helper.get_pr_complexity_analysis().head(5)
        for _, row in complexity.iterrows():
            print(f"   {row['commit_range']}, {row['file_range']}: {row['pr_count']:,} PRs")
        
        print("\n" + "="*60)
        print("Use the AIDevQueryHelper class to run custom queries!")
        print("Example: helper.get_pr_with_commits(pr_id) to see full PR details")
        
    finally:
        helper.close()

if __name__ == "__main__":
    # Change to the directory containing the script
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    main()