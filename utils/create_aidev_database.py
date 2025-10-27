#!/usr/bin/env python3
"""
Script to import AI pull requests and PR commit details, associate them,
and export to SQLite database with proper relationships.
"""

import pandas as pd
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PRDataProcessor:
    def __init__(self, data_dir: str = "data", db_name: str = "aidev_analysis.db"):
        self.data_dir = data_dir
        self.db_name = db_name
        self.conn = None
        
    def setup_database(self):
        """Create SQLite database and tables with proper schema."""
        logger.info(f"Setting up SQLite database: {self.db_name}")
        
        # Remove existing database if it exists
        if Path(self.db_name).exists():
            Path(self.db_name).unlink()
            logger.info("Removed existing database")
        
        self.conn = sqlite3.connect(self.db_name)
        cursor = self.conn.cursor()
        
        # Create pull_requests table
        cursor.execute("""
        CREATE TABLE pull_requests (
            id INTEGER PRIMARY KEY,
            number INTEGER,
            title TEXT,
            body TEXT,
            agent TEXT,
            user_id INTEGER,
            user TEXT,
            state TEXT,
            created_at TEXT,
            closed_at TEXT,
            merged_at TEXT,
            repo_id INTEGER,
            repo_url TEXT,
            html_url TEXT,
            commit_count INTEGER DEFAULT 0,
            file_changes_count INTEGER DEFAULT 0
        )
        """)
        
        # Create commits table
        cursor.execute("""
        CREATE TABLE commits (
            sha TEXT PRIMARY KEY,
            pr_id INTEGER,
            author TEXT,
            committer TEXT,
            message TEXT,
            commit_stats_total INTEGER,
            commit_stats_additions INTEGER,
            commit_stats_deletions INTEGER,
            FOREIGN KEY (pr_id) REFERENCES pull_requests (id)
        )
        """)
        
        # Create file_changes table
        cursor.execute("""
        CREATE TABLE file_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            commit_sha TEXT,
            pr_id INTEGER,
            filename TEXT,
            status TEXT,
            additions INTEGER,
            deletions INTEGER,
            changes INTEGER,
            patch TEXT,
            FOREIGN KEY (commit_sha) REFERENCES commits (sha),
            FOREIGN KEY (pr_id) REFERENCES pull_requests (id)
        )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX idx_commits_pr_id ON commits (pr_id)")
        cursor.execute("CREATE INDEX idx_file_changes_pr_id ON file_changes (pr_id)")
        cursor.execute("CREATE INDEX idx_file_changes_commit_sha ON file_changes (commit_sha)")
        cursor.execute("CREATE INDEX idx_pull_requests_agent ON pull_requests (agent)")
        cursor.execute("CREATE INDEX idx_pull_requests_state ON pull_requests (state)")
        
        self.conn.commit()
        logger.info("Database schema created successfully")
    
    def import_pull_requests(self) -> Dict[int, dict]:
        """Import AI pull requests and return a mapping for reference."""
        logger.info("Importing AI pull requests...")
        
        ai_pr_file = f"{self.data_dir}/AI_pull_request.csv"
        
        # Read in chunks to handle large file
        chunk_size = 10000
        pr_count = 0
        pr_mapping = {}
        
        cursor = self.conn.cursor()
        
        for chunk_num, chunk in enumerate(pd.read_csv(ai_pr_file, chunksize=chunk_size), 1):
            # Handle NaN values
            chunk = chunk.fillna('')
            
            # Helper function to safely convert to int
            def safe_int(value):
                if pd.isna(value) or value == '' or value is None:
                    return None
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return None
            
            # Convert to list of tuples for bulk insert
            pr_data = []
            for _, row in chunk.iterrows():
                pr_tuple = (
                    safe_int(row['id']),
                    safe_int(row['number']),
                    str(row['title']),
                    str(row['body']),
                    str(row['agent']),
                    safe_int(row['user_id']),
                    str(row['user']),
                    str(row['state']),
                    str(row['created_at']),
                    str(row['closed_at']),
                    str(row['merged_at']),
                    safe_int(row['repo_id']),
                    str(row['repo_url']),
                    str(row['html_url'])
                )
                pr_data.append(pr_tuple)
                pr_mapping[safe_int(row['id'])] = {
                    'number': safe_int(row['number']),
                    'title': str(row['title']),
                    'repo_url': str(row['repo_url'])
                }
            
            # Bulk insert
            cursor.executemany("""
                INSERT INTO pull_requests 
                (id, number, title, body, agent, user_id, user, state, created_at, 
                 closed_at, merged_at, repo_id, repo_url, html_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, pr_data)
            
            pr_count += len(chunk)
            
            if chunk_num % 10 == 0:
                logger.info(f"  Imported {pr_count:,} pull requests...")
                self.conn.commit()
        
        self.conn.commit()
        logger.info(f"Imported {pr_count:,} AI pull requests successfully")
        return pr_mapping
    
    def import_commit_details(self, pr_mapping: Dict[int, dict]):
        """Import PR commit details and associate with pull requests."""
        logger.info("Importing PR commit details...")
        
        commit_details_file = f"{self.data_dir}/pr_commit_details.csv"
        
        chunk_size = 50000
        commit_count = 0
        file_changes_count = 0
        pr_commit_stats = {}  # Track stats per PR
        
        cursor = self.conn.cursor()
        
        for chunk_num, chunk in enumerate(pd.read_csv(commit_details_file, chunksize=chunk_size), 1):
            # Handle NaN values
            chunk = chunk.fillna('')
            
            # Separate commits and file changes
            commits_data = []
            file_changes_data = []
            
            # Group by commit SHA to avoid duplicates in commits table
            unique_commits = chunk.drop_duplicates(subset=['sha'])
            
            for _, row in unique_commits.iterrows():
                # Helper function to safely convert to int
                def safe_int(value):
                    if pd.isna(value) or value == '' or value is None:
                        return 0
                    try:
                        return int(float(value))
                    except (ValueError, TypeError):
                        return 0
                
                pr_id = safe_int(row['pr_id'])
                
                # Only process if PR exists in our mapping and pr_id is valid
                if pr_id > 0 and pr_id in pr_mapping:
                    # Helper function to safely convert to int
                    def safe_int(value):
                        if pd.isna(value) or value == '' or value is None:
                            return 0
                        try:
                            return int(float(value))
                        except (ValueError, TypeError):
                            return 0
                    
                    commit_tuple = (
                        str(row['sha']),
                        pr_id,
                        str(row['author']),
                        str(row['committer']),
                        str(row['message']),
                        safe_int(row['commit_stats_total']),
                        safe_int(row['commit_stats_additions']),
                        safe_int(row['commit_stats_deletions'])
                    )
                    commits_data.append(commit_tuple)
                    
                    # Track PR stats
                    if pr_id not in pr_commit_stats:
                        pr_commit_stats[pr_id] = {'commits': set(), 'file_changes': 0}
                    pr_commit_stats[pr_id]['commits'].add(str(row['sha']))
            
            # Process all file changes
            for _, row in chunk.iterrows():
                # Helper function to safely convert to int (reuse)
                def safe_int(value):
                    if pd.isna(value) or value == '' or value is None:
                        return 0
                    try:
                        return int(float(value))
                    except (ValueError, TypeError):
                        return 0
                
                pr_id = safe_int(row['pr_id'])
                
                if pr_id > 0 and pr_id in pr_mapping:
                    # Helper function to safely convert to int (reuse the same function)
                    def safe_int(value):
                        if pd.isna(value) or value == '' or value is None:
                            return 0
                        try:
                            return int(float(value))
                        except (ValueError, TypeError):
                            return 0
                    
                    file_change_tuple = (
                        str(row['sha']),
                        pr_id,
                        str(row['filename']),
                        str(row['status']),
                        safe_int(row['additions']),
                        safe_int(row['deletions']),
                        safe_int(row['changes']),
                        str(row['patch'])
                    )
                    file_changes_data.append(file_change_tuple)
                    
                    # Track file changes per PR
                    if pr_id in pr_commit_stats:
                        pr_commit_stats[pr_id]['file_changes'] += 1
            
            # Bulk insert commits (with OR IGNORE to handle duplicates)
            if commits_data:
                cursor.executemany("""
                    INSERT OR IGNORE INTO commits 
                    (sha, pr_id, author, committer, message, commit_stats_total, 
                     commit_stats_additions, commit_stats_deletions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, commits_data)
                commit_count += len(commits_data)
            
            # Bulk insert file changes
            if file_changes_data:
                cursor.executemany("""
                    INSERT INTO file_changes 
                    (commit_sha, pr_id, filename, status, additions, deletions, changes, patch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, file_changes_data)
                file_changes_count += len(file_changes_data)
            
            if chunk_num % 10 == 0:
                logger.info(f"  Processed {chunk_num * chunk_size:,} rows...")
                logger.info(f"    Commits: {commit_count:,}, File changes: {file_changes_count:,}")
                self.conn.commit()
        
        # Update PR statistics
        logger.info("Updating pull request statistics...")
        for pr_id, stats in pr_commit_stats.items():
            cursor.execute("""
                UPDATE pull_requests 
                SET commit_count = ?, file_changes_count = ?
                WHERE id = ?
            """, (len(stats['commits']), stats['file_changes'], pr_id))
        
        self.conn.commit()
        logger.info(f"Imported {commit_count:,} commits and {file_changes_count:,} file changes")
        return commit_count, file_changes_count
    
    def generate_summary_stats(self):
        """Generate and display summary statistics."""
        logger.info("Generating summary statistics...")
        
        cursor = self.conn.cursor()
        
        # Pull request stats
        cursor.execute("SELECT COUNT(*) FROM pull_requests")
        total_prs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM pull_requests WHERE state = 'merged'")
        merged_prs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM commits")
        total_commits = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM file_changes")
        total_file_changes = cursor.fetchone()[0]
        
        # Top repositories by PR count
        cursor.execute("""
            SELECT repo_url, COUNT(*) as pr_count 
            FROM pull_requests 
            GROUP BY repo_url 
            ORDER BY pr_count DESC 
            LIMIT 5
        """)
        top_repos = cursor.fetchall()
        
        # Average stats per PR
        cursor.execute("""
            SELECT 
                AVG(commit_count) as avg_commits,
                AVG(file_changes_count) as avg_file_changes
            FROM pull_requests 
            WHERE commit_count > 0
        """)
        avg_stats = cursor.fetchone()
        
        # File extension analysis
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN filename LIKE '%.py' THEN 'Python'
                    WHEN filename LIKE '%.js' THEN 'JavaScript'
                    WHEN filename LIKE '%.ts' THEN 'TypeScript'
                    WHEN filename LIKE '%.java' THEN 'Java'
                    WHEN filename LIKE '%.cpp' OR filename LIKE '%.c' THEN 'C/C++'
                    WHEN filename LIKE '%.go' THEN 'Go'
                    WHEN filename LIKE '%.rs' THEN 'Rust'
                    WHEN filename LIKE '%.md' THEN 'Markdown'
                    WHEN filename LIKE '%.json' THEN 'JSON'
                    WHEN filename LIKE '%.yml' OR filename LIKE '%.yaml' THEN 'YAML'
                    ELSE 'Other'
                END as file_type,
                COUNT(*) as change_count
            FROM file_changes 
            GROUP BY file_type 
            ORDER BY change_count DESC
            LIMIT 10
        """)
        file_types = cursor.fetchall()
        
        print("\n" + "="*60)
        print("DATABASE SUMMARY STATISTICS")
        print("="*60)
        print(f"Total Pull Requests: {total_prs:,}")
        print(f"Merged Pull Requests: {merged_prs:,} ({merged_prs/total_prs*100:.1f}%)")
        print(f"Total Commits: {total_commits:,}")
        print(f"Total File Changes: {total_file_changes:,}")
        
        if avg_stats[0]:
            print(f"\nAverage per PR:")
            print(f"  Commits: {avg_stats[0]:.1f}")
            print(f"  File Changes: {avg_stats[1]:.1f}")
        
        print(f"\nTop 5 Repositories by PR Count:")
        for repo, count in top_repos:
            repo_name = repo.split('/')[-1] if repo else 'Unknown'
            print(f"  {repo_name}: {count:,} PRs")
        
        print(f"\nFile Type Distribution:")
        for file_type, count in file_types:
            print(f"  {file_type}: {count:,} changes")
        
        print("="*60)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info(f"Database saved as: {self.db_name}")

def main():
    """Main execution function."""
    start_time = time.time()
    
    print("AI Development Analysis - Database Creation")
    print("="*50)
    
    # Initialize processor
    processor = PRDataProcessor(data_dir="data", db_name="aidev_analysis.db")
    
    try:
        # Set up database
        processor.setup_database()
        
        # Import pull requests
        pr_mapping = processor.import_pull_requests()
        
        # Import commit details and associate with PRs
        commit_count, file_changes_count = processor.import_commit_details(pr_mapping)
        
        # Generate summary statistics
        processor.generate_summary_stats()
        
        total_time = time.time() - start_time
        print(f"\nProcessing completed in {total_time:.2f} seconds")
        print(f"Database file: {processor.db_name}")
        print(f"Database size: {Path(processor.db_name).stat().st_size / (1024*1024):.1f} MB")
        
        # Save processing log
        with open("database_creation_log.txt", "w") as f:
            f.write(f"AI Development Analysis Database Creation Log\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Creation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {total_time:.2f} seconds\n")
            f.write(f"Database File: {processor.db_name}\n")
            f.write(f"Database Size: {Path(processor.db_name).stat().st_size / (1024*1024):.1f} MB\n\n")
            f.write(f"Data Imported:\n")
            f.write(f"- Pull Requests: {len(pr_mapping):,}\n")
            f.write(f"- Commits: {commit_count:,}\n")
            f.write(f"- File Changes: {file_changes_count:,}\n")
        
        print("\nProcessing log saved to: database_creation_log.txt")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        processor.close()

if __name__ == "__main__":
    # Change to the directory containing the script
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    main()