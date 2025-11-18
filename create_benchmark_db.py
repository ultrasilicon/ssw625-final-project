#!/usr/bin/env python3
"""
Script to create benchmark.db with 100 random sample PRs for each agent.
Each agent gets its own table with merged pull request and commit details data.
"""

import sqlite3
import random
from pathlib import Path
import pandas as pd

class BenchmarkCreator:
    def __init__(self, source_db="research_pr_analysis.db", target_db="benchmark.db"):
        self.source_db = source_db
        self.target_db = target_db
        self.source_conn = None
        self.target_conn = None
        
    def connect(self):
        """Connect to source and target databases."""
        if not Path(self.source_db).exists():
            raise FileNotFoundError(f"Source database {self.source_db} not found!")
        
        self.source_conn = sqlite3.connect(self.source_db)
        
        # Remove existing benchmark.db if it exists
        if Path(self.target_db).exists():
            Path(self.target_db).unlink()
            print(f"Removed existing {self.target_db}")
        
        self.target_conn = sqlite3.connect(self.target_db)
        print(f"Connected to databases")
        
    def get_agents(self):
        """Get all unique agents from the source database."""
        cursor = self.source_conn.cursor()
        cursor.execute("SELECT DISTINCT agent FROM pull_requests ORDER BY agent")
        agents = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(agents)} agents: {agents}")
        return agents
    
    def get_random_prs_for_agent(self, agent, sample_size=100):
        """Get random PR IDs for a specific agent that have commit details."""
        cursor = self.source_conn.cursor()
        
        # Get PR IDs for this agent that have commit details
        query = """
        SELECT DISTINCT pr.pr_id 
        FROM pull_requests pr 
        INNER JOIN commit_details cd ON pr.pr_id = cd.pr_id 
        WHERE pr.agent = ?
        """
        cursor.execute(query, (agent,))
        available_pr_ids = [row[0] for row in cursor.fetchall()]
        
        print(f"  {agent}: {len(available_pr_ids)} PRs with commit details available")
        
        # Sample random PRs (or all if less than sample_size)
        if len(available_pr_ids) <= sample_size:
            selected_pr_ids = available_pr_ids
            print(f"  {agent}: Selected all {len(selected_pr_ids)} PRs (less than {sample_size})")
        else:
            selected_pr_ids = random.sample(available_pr_ids, sample_size)
            print(f"  {agent}: Randomly selected {len(selected_pr_ids)} PRs")
        
        return selected_pr_ids
    
    def create_agent_table(self, agent, pr_ids):
        """Create a table for the agent with merged PR and commit details."""
        # Clean agent name for table name
        table_name = "".join(c for c in agent if c.isalnum() or c in ('_',)).lower()
        if not table_name:
            table_name = "unknown_agent"
        
        cursor = self.target_conn.cursor()
        
        # Create table with merged schema
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            -- Pull Request fields (14 columns)
            pr_id INTEGER,
            pr_number INTEGER,
            pr_title TEXT,
            pr_body TEXT,
            pr_agent TEXT,
            pr_user_id INTEGER,
            pr_user TEXT,
            pr_state TEXT,
            pr_created_at TEXT,
            pr_closed_at TEXT,
            pr_merged_at TEXT,
            pr_repo_id INTEGER,
            pr_repo_url TEXT,
            pr_html_url TEXT,
            
            -- Commit Details fields (14 columns, excluding pr_id since it's same as above)
            commit_detail_id INTEGER,
            commit_sha TEXT,
            commit_author TEXT,
            commit_committer TEXT,
            commit_message TEXT,
            commit_stats_total INTEGER,
            commit_stats_additions INTEGER,
            commit_stats_deletions INTEGER,
            file_filename TEXT,
            file_status TEXT,
            file_additions INTEGER,
            file_deletions INTEGER,
            file_changes INTEGER,
            file_patch TEXT
        )
        """
        
        cursor.execute(create_table_sql)
        print(f"  Created table: {table_name}")
        
        # Get merged data for these PR IDs
        source_cursor = self.source_conn.cursor()
        
        placeholders = ','.join('?' * len(pr_ids))
        merge_query = f"""
        SELECT 
            -- Pull Request fields
            pr.pr_id,
            pr.number,
            pr.title,
            pr.body,
            pr.agent,
            pr.user_id,
            pr.user,
            pr.state,
            pr.created_at,
            pr.closed_at,
            pr.merged_at,
            pr.repo_id,
            pr.repo_url,
            pr.html_url,
            
            -- Commit Details fields
            cd.id,
            cd.sha,
            cd.author,
            cd.committer,
            cd.message,
            cd.commit_stats_total,
            cd.commit_stats_additions,
            cd.commit_stats_deletions,
            
            -- File Details fields
            cd.filename,
            cd.status,
            cd.additions,
            cd.deletions,
            cd.changes,
            cd.patch
        FROM pull_requests pr
        INNER JOIN commit_details cd ON pr.pr_id = cd.pr_id
        WHERE pr.pr_id IN ({placeholders})
        ORDER BY pr.pr_id, cd.id
        """
        
        source_cursor.execute(merge_query, pr_ids)
        merged_data = source_cursor.fetchall()
        
        # Insert merged data
        insert_placeholders = ','.join('?' * 28)  # 28 columns total (14 + 14)
        insert_sql = f"INSERT INTO {table_name} VALUES ({insert_placeholders})"
        
        cursor.executemany(insert_sql, merged_data)
        self.target_conn.commit()
        
        print(f"  Inserted {len(merged_data)} merged records for {len(pr_ids)} PRs")
        
        return table_name, len(merged_data)
    
    def create_benchmark_summary_table(self, agent_stats):
        """Create a summary table with statistics about the benchmark data."""
        cursor = self.target_conn.cursor()
        
        cursor.execute("""
        CREATE TABLE benchmark_summary (
            agent TEXT,
            table_name TEXT,
            pr_count INTEGER,
            record_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        for agent, (table_name, record_count, pr_count) in agent_stats.items():
            cursor.execute("""
            INSERT INTO benchmark_summary (agent, table_name, pr_count, record_count)
            VALUES (?, ?, ?, ?)
            """, (agent, table_name, pr_count, record_count))
        
        self.target_conn.commit()
        print("Created benchmark_summary table")
    
    def create_benchmark(self, sample_size=100):
        """Create the benchmark database."""
        print("="*80)
        print("CREATING BENCHMARK DATABASE")
        print("="*80)
        
        # Set random seed for reproducibility
        random.seed(42)
        print(f"Set random seed to 42 for reproducibility")
        
        agents = self.get_agents()
        agent_stats = {}
        
        for agent in agents:
            print(f"\nProcessing agent: {agent}")
            print("-" * 40)
            
            # Get random PR IDs for this agent
            pr_ids = self.get_random_prs_for_agent(agent, sample_size)
            
            if pr_ids:
                # Create table with merged data
                table_name, record_count = self.create_agent_table(agent, pr_ids)
                agent_stats[agent] = (table_name, record_count, len(pr_ids))
            else:
                print(f"  {agent}: No PRs with commit details found")
                agent_stats[agent] = (None, 0, 0)
        
        # Create summary table
        print(f"\nCreating summary table...")
        self.create_benchmark_summary_table(agent_stats)
        
        return agent_stats
    
    def print_summary(self, agent_stats):
        """Print summary of created benchmark database."""
        print("\n" + "="*80)
        print("BENCHMARK DATABASE SUMMARY")
        print("="*80)
        
        total_prs = 0
        total_records = 0
        
        print(f"Database: {self.target_db}")
        print(f"Sample size per agent: 100 (or all available if less)")
        print("\nAgent Tables Created:")
        
        for agent, (table_name, record_count, pr_count) in agent_stats.items():
            if table_name:
                print(f"  {agent}:")
                print(f"    Table: {table_name}")
                print(f"    PRs: {pr_count}")
                print(f"    Records: {record_count:,}")
                total_prs += pr_count
                total_records += record_count
            else:
                print(f"  {agent}: No data")
        
        print(f"\nTotal PRs in benchmark: {total_prs}")
        print(f"Total records in benchmark: {total_records:,}")
        
        # Get file size
        if Path(self.target_db).exists():
            file_size = Path(self.target_db).stat().st_size / (1024 * 1024)  # MB
            print(f"Database size: {file_size:.1f} MB")
    
    def close(self):
        """Close database connections."""
        if self.source_conn:
            self.source_conn.close()
        if self.target_conn:
            self.target_conn.close()

def main():
    """Main execution function."""
    print("Benchmark Database Creator")
    print("="*80)
    
    creator = BenchmarkCreator("research_pr_analysis.db", "benchmark.db")
    
    try:
        # Connect to databases
        creator.connect()
        
        # Create benchmark database
        agent_stats = creator.create_benchmark(sample_size=100)
        
        # Print summary
        creator.print_summary(agent_stats)
        
        print(f"\n✅ Benchmark database created successfully!")
        print(f"File: {creator.target_db}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
        
    finally:
        creator.close()

if __name__ == "__main__":
    main()