#!/usr/bin/env python3
"""
Script to analyze research_pr_analysis.db:
1. Print database headers/schema
2. Count different types of agents
3. Split database by agent into separate database files
"""

import sqlite3
import os
from pathlib import Path
import pandas as pd

class DatabaseAnalyzer:
    def __init__(self, db_path="research_pr_analysis.db"):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to the database."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database {self.db_path} not found!")
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")
        
    def print_headers(self):
        """Print database schema and headers."""
        print("\n" + "="*80)
        print("DATABASE SCHEMA AND HEADERS")
        print("="*80)
        
        cursor = self.conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"Database: {self.db_path}")
        print(f"Total Tables: {len(tables)}")
        print()
        
        for table in tables:
            table_name = table[0]
            print(f"TABLE: {table_name}")
            print("-" * 60)
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print("Columns:")
            for col in columns:
                col_id, name, data_type, not_null, default_val, primary_key = col
                pk_indicator = " (PRIMARY KEY)" if primary_key else ""
                null_indicator = " NOT NULL" if not_null else ""
                default_indicator = f" DEFAULT {default_val}" if default_val else ""
                print(f"  {name}: {data_type}{pk_indicator}{null_indicator}{default_indicator}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"Row Count: {count:,}")
            print()
    
    def count_agents(self):
        """Count and analyze different types of agents."""
        print("="*80)
        print("AGENT ANALYSIS")
        print("="*80)
        
        cursor = self.conn.cursor()
        
        # Check if agent column exists in pull_requests table
        cursor.execute("PRAGMA table_info(pull_requests)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'agent' not in columns:
            print("No 'agent' column found in pull_requests table!")
            print("Available columns:", columns)
            return {}
        
        # Count agents
        cursor.execute("SELECT agent, COUNT(*) as count FROM pull_requests GROUP BY agent ORDER BY count DESC")
        agent_counts = cursor.fetchall()
        
        print(f"Total unique agents: {len(agent_counts)}")
        print("\nAgent distribution:")
        
        agent_dict = {}
        for agent, count in agent_counts:
            agent_dict[agent] = count
            print(f"  {agent}: {count:,} PRs")
        
        total_prs = sum(agent_dict.values())
        print(f"\nTotal PRs: {total_prs:,}")
        
        return agent_dict
    
    def create_agent_database(self, agent_name, output_dir="agent_databases"):
        """Create a separate database for a specific agent."""
        # Clean agent name for filename
        safe_agent_name = "".join(c for c in agent_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_agent_name = safe_agent_name.replace(' ', '_')
        
        if not safe_agent_name:
            safe_agent_name = "unknown_agent"
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create new database file
        agent_db_path = f"{output_dir}/{safe_agent_name}.db"
        
        # Remove existing file if it exists
        if Path(agent_db_path).exists():
            Path(agent_db_path).unlink()
            
        agent_conn = sqlite3.connect(agent_db_path)
        agent_cursor = agent_conn.cursor()
        
        # Get original schema and create tables
        cursor = self.conn.cursor()
        
        # Get all table creation statements
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
        table_schemas = cursor.fetchall()
        
        for schema in table_schemas:
            if schema[0]:  # Skip None values
                try:
                    agent_cursor.execute(schema[0])
                except sqlite3.Error as e:
                    print(f"    Warning: Could not create table from schema: {e}")
                    continue
        
        # Copy data for this agent
        # First, get all PR IDs for this agent
        cursor.execute("SELECT pr_id FROM pull_requests WHERE agent = ?", (agent_name,))
        pr_ids = [row[0] for row in cursor.fetchall()]
        
        if pr_ids:
            # Copy pull requests
            placeholders = ','.join('?' * len(pr_ids))
            cursor.execute(f"SELECT * FROM pull_requests WHERE pr_id IN ({placeholders})", pr_ids)
            pr_data = cursor.fetchall()
            
            # Get column names for pull_requests
            cursor.execute("PRAGMA table_info(pull_requests)")
            pr_columns = [col[1] for col in cursor.fetchall()]
            insert_placeholders = ','.join('?' * len(pr_columns))
            
            agent_cursor.executemany(f"INSERT INTO pull_requests VALUES ({insert_placeholders})", pr_data)
            
            # Copy commit_details
            cursor.execute(f"SELECT * FROM commit_details WHERE pr_id IN ({placeholders})", pr_ids)
            commit_data = cursor.fetchall()
            
            if commit_data:
                cursor.execute("PRAGMA table_info(commit_details)")
                commit_columns = [col[1] for col in cursor.fetchall()]
                insert_placeholders = ','.join('?' * len(commit_columns))
                
                agent_cursor.executemany(f"INSERT INTO commit_details VALUES ({insert_placeholders})", commit_data)
            
            # Copy benchmark_scores if they exist
            cursor.execute(f"SELECT * FROM benchmark_scores WHERE pr_id IN ({placeholders})", pr_ids)
            benchmark_data = cursor.fetchall()
            
            if benchmark_data:
                cursor.execute("PRAGMA table_info(benchmark_scores)")
                benchmark_columns = [col[1] for col in cursor.fetchall()]
                insert_placeholders = ','.join('?' * len(benchmark_columns))
                
                agent_cursor.executemany(f"INSERT INTO benchmark_scores VALUES ({insert_placeholders})", benchmark_data)
        
        agent_conn.commit()
        agent_conn.close()
        
        # Get file size
        file_size = Path(agent_db_path).stat().st_size / (1024 * 1024)  # MB
        
        print(f"  Created: {agent_db_path} ({file_size:.1f} MB) - {len(pr_ids):,} PRs")
        
        return agent_db_path
    
    def split_database_by_agents(self, agent_dict):
        """Split the database into separate files for each agent."""
        print("\n" + "="*80)
        print("SPLITTING DATABASE BY AGENTS")
        print("="*80)
        
        output_dir = "agent_databases"
        created_dbs = []
        
        for agent_name in agent_dict.keys():
            try:
                db_path = self.create_agent_database(agent_name, output_dir)
                created_dbs.append(db_path)
            except Exception as e:
                print(f"  Error creating database for agent '{agent_name}': {e}")
        
        print(f"\nSplit complete! Created {len(created_dbs)} agent databases in '{output_dir}/' directory")
        return created_dbs
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    """Main execution function."""
    print("Database Analysis and Agent Splitting Tool")
    print("="*80)
    
    analyzer = DatabaseAnalyzer("research_pr_analysis.db")
    
    try:
        # Connect to database
        analyzer.connect()
        
        # Print headers and schema
        analyzer.print_headers()
        
        # Count and analyze agents
        agent_dict = analyzer.count_agents()
        
        if agent_dict:
            # Split database by agents
            created_dbs = analyzer.split_database_by_agents(agent_dict)
            
            # Summary
            print("\n" + "="*80)
            print("SUMMARY")
            print("="*80)
            print(f"Original database: research_pr_analysis.db")
            print(f"Total agents: {len(agent_dict)}")
            print(f"Created agent databases: {len(created_dbs)}")
            print("\nAgent databases created:")
            
            for db_path in created_dbs:
                db_name = Path(db_path).name
                agent_name = db_name.replace('.db', '').replace('_', ' ')
                print(f"  {db_name}")
        else:
            print("No agents found to split database.")
            
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()