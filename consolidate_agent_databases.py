#!/usr/bin/env python3
"""
Script to consolidate all agent databases into one unified database
with agent-prefixed table names.
"""

import sqlite3
import os
from pathlib import Path

class DatabaseConsolidator:
    def __init__(self, agent_db_dir="agent_databases", output_db="unified_agents.db"):
        self.agent_db_dir = agent_db_dir
        self.output_db = output_db
        self.unified_conn = None
        
    def create_unified_database(self):
        """Create the unified database and connection."""
        # Remove existing unified database if it exists
        if Path(self.output_db).exists():
            Path(self.output_db).unlink()
            print(f"Removed existing {self.output_db}")
        
        self.unified_conn = sqlite3.connect(self.output_db)
        print(f"Created unified database: {self.output_db}")
    
    def get_agent_databases(self):
        """Get list of all agent database files."""
        agent_db_path = Path(self.agent_db_dir)
        if not agent_db_path.exists():
            raise FileNotFoundError(f"Agent databases directory {self.agent_db_dir} not found!")
        
        db_files = list(agent_db_path.glob("*.db"))
        return sorted(db_files)
    
    def get_table_schema(self, cursor, table_name):
        """Get the CREATE TABLE statement for a table."""
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def consolidate_agent_database(self, db_file):
        """Consolidate one agent database into the unified database."""
        agent_name = db_file.stem
        print(f"\nProcessing {agent_name} database...")
        
        # Connect to agent database
        agent_conn = sqlite3.connect(str(db_file))
        agent_cursor = agent_conn.cursor()
        
        # Get all tables in the agent database
        agent_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
        tables = agent_cursor.fetchall()
        
        unified_cursor = self.unified_conn.cursor()
        
        for table_tuple in tables:
            table_name = table_tuple[0]
            new_table_name = f"{agent_name}_{table_name}"
            
            print(f"  Copying {table_name} -> {new_table_name}")
            
            # Get table schema
            schema = self.get_table_schema(agent_cursor, table_name)
            if not schema:
                print(f"    Warning: Could not get schema for {table_name}")
                continue
            
            # Modify schema to use new table name
            new_schema = schema.replace(f"CREATE TABLE {table_name}", f"CREATE TABLE {new_table_name}")
            
            # Create table in unified database
            try:
                unified_cursor.execute(new_schema)
            except sqlite3.Error as e:
                print(f"    Error creating table {new_table_name}: {e}")
                continue
            
            # Get column names
            agent_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in agent_cursor.fetchall()]
            
            # Copy data
            agent_cursor.execute(f"SELECT * FROM {table_name}")
            rows = agent_cursor.fetchall()
            
            if rows:
                placeholders = ','.join('?' * len(columns))
                unified_cursor.executemany(
                    f"INSERT INTO {new_table_name} VALUES ({placeholders})",
                    rows
                )
                
                print(f"    Copied {len(rows):,} rows")
            else:
                print(f"    Table {table_name} is empty")
        
        agent_conn.close()
    
    def create_summary_view(self):
        """Create a view that shows summary statistics from all agent tables."""
        unified_cursor = self.unified_conn.cursor()
        
        # Get all pull_requests tables
        unified_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_pull_requests'")
        pr_tables = [row[0] for row in unified_cursor.fetchall()]
        
        if pr_tables:
            # Create a union view of all pull requests with agent identifier
            union_parts = []
            for table in pr_tables:
                agent_name = table.replace('_pull_requests', '')
                union_parts.append(f"SELECT '{agent_name}' as source_agent, * FROM {table}")
            
            union_query = " UNION ALL ".join(union_parts)
            
            view_sql = f"""
            CREATE VIEW all_pull_requests AS
            {union_query}
            """
            
            try:
                unified_cursor.execute(view_sql)
                print(f"\nCreated view 'all_pull_requests' combining {len(pr_tables)} agent tables")
            except sqlite3.Error as e:
                print(f"Warning: Could not create summary view: {e}")
    
    def generate_statistics(self):
        """Generate statistics for the unified database."""
        print(f"\n{'='*80}")
        print("UNIFIED DATABASE STATISTICS")
        print(f"{'='*80}")
        
        unified_cursor = self.unified_conn.cursor()
        
        # Get all tables
        unified_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = unified_cursor.fetchall()
        
        total_rows = 0
        
        for table_tuple in tables:
            table_name = table_tuple[0]
            
            # Get row count
            unified_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = unified_cursor.fetchone()[0]
            total_rows += row_count
            
            # Extract agent and table type
            if '_' in table_name:
                agent, table_type = table_name.split('_', 1)
                print(f"  {agent:15} | {table_type:20} | {row_count:>8,} rows")
            else:
                print(f"  {'':15} | {table_name:20} | {row_count:>8,} rows")
        
        print(f"\nTotal rows across all tables: {total_rows:,}")
        
        # Get file size
        file_size = Path(self.output_db).stat().st_size / (1024 * 1024)
        print(f"Database file size: {file_size:.1f} MB")
        
        # Agent summary
        print(f"\nAgent Summary:")
        unified_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_pull_requests'")
        pr_tables = unified_cursor.fetchall()
        
        for table_tuple in pr_tables:
            table_name = table_tuple[0]
            agent = table_name.replace('_pull_requests', '')
            
            unified_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            pr_count = unified_cursor.fetchone()[0]
            
            # Try to get commit details count
            commit_table = f"{agent}_commit_details"
            try:
                unified_cursor.execute(f"SELECT COUNT(*) FROM {commit_table}")
                commit_count = unified_cursor.fetchone()[0]
                print(f"  {agent:15}: {pr_count:>6,} PRs, {commit_count:>8,} commit details")
            except sqlite3.Error:
                print(f"  {agent:15}: {pr_count:>6,} PRs, no commit details table")
    
    def close(self):
        """Close database connection."""
        if self.unified_conn:
            self.unified_conn.commit()
            self.unified_conn.close()

def main():
    """Main execution function."""
    print("Database Consolidation Tool")
    print("="*50)
    
    consolidator = DatabaseConsolidator()
    
    try:
        # Create unified database
        consolidator.create_unified_database()
        
        # Get all agent databases
        agent_dbs = consolidator.get_agent_databases()
        
        if not agent_dbs:
            print("No agent databases found in agent_databases/ directory!")
            return
        
        print(f"Found {len(agent_dbs)} agent databases:")
        for db in agent_dbs:
            print(f"  - {db.name}")
        
        # Consolidate each agent database
        for db_file in agent_dbs:
            consolidator.consolidate_agent_database(db_file)
        
        # Create summary view
        consolidator.create_summary_view()
        
        # Generate statistics
        consolidator.generate_statistics()
        
        print(f"\n{'='*80}")
        print("CONSOLIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"Unified database: {consolidator.output_db}")
        print("Original agent databases preserved in agent_databases/ directory")
        
        # Show table list
        unified_cursor = consolidator.unified_conn.cursor()
        unified_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in unified_cursor.fetchall()]
        
        print(f"\nTables in unified database ({len(tables)} total):")
        for table in tables:
            print(f"  - {table}")
        
        # Show views
        unified_cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = [row[0] for row in unified_cursor.fetchall()]
        
        if views:
            print(f"\nViews created ({len(views)} total):")
            for view in views:
                print(f"  - {view}")
        
    except Exception as e:
        print(f"Error during consolidation: {e}")
        raise
    finally:
        consolidator.close()

if __name__ == "__main__":
    main()