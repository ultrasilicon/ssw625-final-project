#!/usr/bin/env python3
"""
Script to create a research database for benchmarking agentic PR descriptions.
Loads data from ai_pull_request.csv and pr_commit_details.csv into SQLite database.
"""

import sqlite3
import pandas as pd
import sys
from pathlib import Path

# Database configuration
DB_PATH = 'research_pr_analysis.db'
DATA_DIR = Path('data')

# CSV file paths
AI_PR_CSV = DATA_DIR / 'ai_pull_request.csv'
PR_COMMIT_DETAILS_CSV = DATA_DIR / 'pr_commit_details.csv'


def create_database_schema(conn):
    """Create the database schema for PR analysis research."""
    cursor = conn.cursor()
    
    # Main Pull Requests table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pull_requests (
        pr_id INTEGER PRIMARY KEY,
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
        html_url TEXT
    )
    ''')
    
    # Commit Details table (linked to PRs via pr_id)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS commit_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sha TEXT,
        pr_id INTEGER,
        author TEXT,
        committer TEXT,
        message TEXT,
        commit_stats_total INTEGER,
        commit_stats_additions INTEGER,
        commit_stats_deletions INTEGER,
        filename TEXT,
        status TEXT,
        additions INTEGER,
        deletions INTEGER,
        changes INTEGER,
        patch TEXT,
        FOREIGN KEY (pr_id) REFERENCES pull_requests(pr_id)
    )
    ''')
    
    # Benchmark Scores table (for future evaluation)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS benchmark_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_id INTEGER,
        llm_model TEXT,
        coverage_score INTEGER CHECK(coverage_score BETWEEN 1 AND 5),
        clarity_score INTEGER CHECK(clarity_score BETWEEN 1 AND 5),
        readability_score INTEGER CHECK(readability_score BETWEEN 1 AND 5),
        tells_why_written_score INTEGER CHECK(tells_why_written_score BETWEEN 1 AND 5),
        tells_changes_made_score INTEGER CHECK(tells_changes_made_score BETWEEN 1 AND 5),
        tells_why_this_way_score INTEGER CHECK(tells_why_this_way_score BETWEEN 1 AND 5),
        covers_all_commits_score INTEGER CHECK(covers_all_commits_score BETWEEN 1 AND 5),
        tells_how_to_test_score INTEGER CHECK(tells_how_to_test_score BETWEEN 1 AND 5),
        notes TEXT,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (pr_id) REFERENCES pull_requests(pr_id)
    )
    ''')
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pr_agent ON pull_requests(agent)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_pr_repo ON pull_requests(repo_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_commit_pr ON commit_details(pr_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_commit_sha ON commit_details(sha)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmark_pr ON benchmark_scores(pr_id)')
    
    conn.commit()
    print("Database schema created successfully")


def load_pull_requests(conn):
    """Load pull requests data from CSV into database."""
    print(f"\nLoading pull requests from {AI_PR_CSV}...")
    
    # Read CSV in chunks to handle large files
    chunk_size = 10000
    total_rows = 0
    
    for chunk in pd.read_csv(AI_PR_CSV, chunksize=chunk_size):
        # Rename 'id' column to 'pr_id' for consistency
        chunk = chunk.rename(columns={'id': 'pr_id'})
        
        # Write to database
        chunk.to_sql('pull_requests', conn, if_exists='append', index=False)
        total_rows += len(chunk)

        
        print(f"  Loaded {total_rows} pull requests...", end='\r')
    
    print(f"\nLoaded {total_rows} pull requests")
    return total_rows


def load_commit_details(conn):
    """Load commit details data from CSV into database."""
    print(f"\nLoading commit details from {PR_COMMIT_DETAILS_CSV}...")
    print("  (This may take a while due to large data size...)")
    
    # Read CSV in chunks to handle large files
    chunk_size = 10000
    total_rows = 0
    

    # Optimize SQLite for bulk insert
    conn.execute('PRAGMA synchronous = OFF')
    conn.execute('PRAGMA journal_mode = MEMORY')
    conn.execute('PRAGMA cache_size = 10000')
    
    try:
        for i, chunk in enumerate(pd.read_csv(PR_COMMIT_DETAILS_CSV, chunksize=chunk_size, low_memory=False)):
            # Write to database without method='multi' to avoid SQL variable limit
            chunk.to_sql('commit_details', conn, if_exists='append', index=False)

            total_rows += len(chunk)
            
            if (i + 1) % 20 == 0:  # Commit every 20 chunks
                conn.commit()
            
            print(f"  Loaded {total_rows:,} commit details...", end='\r')
        
        conn.commit()
        print(f"\nLoaded {total_rows:,} commit details")
        
    finally:
        # Restore normal settings
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA journal_mode = DELETE')
    
    return total_rows


def create_combined_view(conn):
    """Create a view that combines PR and commit information."""
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE VIEW IF NOT EXISTS pr_with_commits AS
    SELECT 
        pr.pr_id,
        pr.agent,
        pr.repo_id,
        pr.title,
        pr.body,
        pr.number,
        pr.state,
        pr.created_at,
        pr.merged_at,
        cd.sha,
        cd.author,
        cd.committer,
        cd.message,
        cd.commit_stats_total,
        cd.commit_stats_additions,
        cd.commit_stats_deletions,
        cd.filename,
        cd.status,
        cd.additions,
        cd.deletions,
        cd.changes,
        cd.patch
    FROM pull_requests pr
    LEFT JOIN commit_details cd ON pr.pr_id = cd.pr_id
    ''')
    
    conn.commit()
    print("Created combined view: pr_with_commits")


def print_database_info(conn):
    """Print information about the database structure and contents."""
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("DATABASE STRUCTURE AND STATISTICS")
    print("="*80)

    
    # Pull Requests table info
    print("\nTABLE: pull_requests")
    print("-" * 80)
    cursor.execute("PRAGMA table_info(pull_requests)")
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  • {col[1]:25s} {col[2]:15s} {'(PRIMARY KEY)' if col[5] else ''}")
    
    cursor.execute("SELECT COUNT(*) FROM pull_requests")
    count = cursor.fetchone()[0]
    print(f"\nTotal PRs: {count:,}")
    
    # Commit Details table info
    print("\nTABLE: commit_details")
    print("-" * 80)
    cursor.execute("PRAGMA table_info(commit_details)")
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  • {col[1]:30s} {col[2]:15s} {'(PRIMARY KEY)' if col[5] else ''}")
    
    cursor.execute("SELECT COUNT(*) FROM commit_details")
    count = cursor.fetchone()[0]
    print(f"\nTotal Commit Details: {count:,}")
    

    # Benchmark Scores table info
    print("\nTABLE: benchmark_scores (for future evaluation)")
    print("-" * 80)
    cursor.execute("PRAGMA table_info(benchmark_scores)")
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  • {col[1]:30s} {col[2]:15s} {'(PRIMARY KEY)' if col[5] else ''}")
    
    # Combined view info
    print("\nVIEW: pr_with_commits (combined PR and commit data)")
    print("-" * 80)
    cursor.execute("PRAGMA table_info(pr_with_commits)")
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  • {col[1]:30s} {col[2]:15s}")
    
    # Sample statistics
    print("\nSAMPLE STATISTICS")
    print("-" * 80)
    
    cursor.execute("""
        SELECT agent, COUNT(*) as count 
        FROM pull_requests 
        GROUP BY agent 
        ORDER BY count DESC
    """)
    print("\nPRs by Agent:")
    for row in cursor.fetchall():
        print(f"  • {row[0]:20s}: {row[1]:,} PRs")

def main():
    """Main execution function."""
    print("="*80)
    print("CREATING RESEARCH DATABASE FOR PR ANALYSIS")
    print("="*80)
    
    # Check if CSV files exist
    if not AI_PR_CSV.exists():
        print(f"Error: {AI_PR_CSV} not found!")
        sys.exit(1)
    
    if not PR_COMMIT_DETAILS_CSV.exists():
        print(f"Error: {PR_COMMIT_DETAILS_CSV} not found!")
        sys.exit(1)
    
    # Create database connection
    print(f"\nCreating database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Create schema
        create_database_schema(conn)
        
        # Load data
        pr_count = load_pull_requests(conn)
        commit_count = load_commit_details(conn)
        
        # Create combined view
        create_combined_view(conn)
        
        # Print database information
        print_database_info(conn)
        
        print("\nDatabase created successfully!")
        print(f"Database file: {DB_PATH}")
        print(f"Total PRs: {pr_count:,}")
        print(f"Total Commit Details: {commit_count:,}")
        
        print("\nUSAGE EXAMPLES:")
        print("-" * 80)
        print("# Query all data for a specific PR:")
        print("  SELECT * FROM pr_with_commits WHERE pr_id = 123456789;")
        print("\n# Get PRs by agent:")
        print("  SELECT * FROM pull_requests WHERE agent = 'cursor';")
        print("\n# Count commits per PR:")
        print("  SELECT pr_id, COUNT(*) as commit_count")
        print("  FROM commit_details GROUP BY pr_id;")
        print("\n# Future: Add benchmark scores:")
        print("  INSERT INTO benchmark_scores (pr_id, llm_model, coverage_score, ...)")
        print("  VALUES (123456789, 'gpt-4', 4, ...);")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()

