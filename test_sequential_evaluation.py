#!/usr/bin/env python3
"""
Quick test of the sequential evaluation approach
"""

import sqlite3
import json
import random
import time
from pathlib import Path
from typing import Dict, List
import threading
from dataclasses import dataclass

@dataclass
class PRData:
    pr_id: int
    title: str
    body: str
    agent: str
    commits: List[Dict]
    commit_messages: List[str]
    file_changes: List[Dict]

@dataclass 
class EvaluationResult:
    pr_id: int
    agent: str
    model: str
    coverage_purpose: int
    coverage_changes: int
    rationale_clarity: int
    commit_coverage: int
    testing_guidance: int
    readability: int
    
    def to_dict(self):
        return {
            'pr_id': self.pr_id,
            'agent': self.agent,
            'model': self.model,
            'coverage_purpose': self.coverage_purpose,
            'coverage_changes': self.coverage_changes,
            'rationale_clarity': self.rationale_clarity,
            'commit_coverage': self.commit_coverage,
            'testing_guidance': self.testing_guidance,
            'readability': self.readability
        }

class TestSequentialEvaluator:
    def __init__(self, db_path="agent_pr.db"):
        self.db_path = db_path
        self.conn = None
        self._lock = threading.Lock()
        
        # Mock models for testing
        self.models = ['Deepseek-V3-0324', 'DeepSeek-V3.1']  # Test with 2 models only
        
    def connect_database(self):
        """Connect to the database."""
        if not Path(self.db_path).exists():
            print(f"Database {self.db_path} not found!")
            return False
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        print(f"Connected to database: {self.db_path}")
        return True
        
    def get_agent_tables(self) -> List[str]:
        """Get list of agent names from pull request tables."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_pull_requests'")
        tables = cursor.fetchall()
        
        agents = []
        for table_tuple in tables:
            table_name = table_tuple[0]
            agent = table_name.replace('_pull_requests', '')
            agents.append(agent)
        
        return sorted(agents)
    
    def get_random_prs_for_agent(self, agent: str, count: int = 5) -> List[PRData]:
        """Get random PRs for a specific agent with their commit details."""
        with self._lock:  # Thread-safe database access
            cursor = self.conn.cursor()
            
            # Get all PR IDs for this agent
            pr_table = f"{agent}_pull_requests"
            commit_table = f"{agent}_commit_details"
            
            cursor.execute(f"SELECT pr_id FROM {pr_table}")
            all_pr_ids = [row[0] for row in cursor.fetchall()]
            
            if len(all_pr_ids) < count:
                print(f"Warning: Only {len(all_pr_ids)} PRs available for {agent}, using all of them")
                selected_pr_ids = all_pr_ids
            else:
                selected_pr_ids = random.sample(all_pr_ids, count)
            
            pr_data_list = []
            
            for pr_id in selected_pr_ids:
                # Get PR details
                cursor.execute(f"""
                    SELECT pr_id, title, body, agent 
                    FROM {pr_table} 
                    WHERE pr_id = ?
                """, (pr_id,))
                pr_row = cursor.fetchone()
                
                if not pr_row:
                    continue
                    
                # Simplified commit data for testing
                pr_data = PRData(
                    pr_id=pr_row[0],
                    title=pr_row[1] or "",
                    body=pr_row[2] or "",
                    agent=pr_row[3],
                    commits=[],
                    commit_messages=[],
                    file_changes=[]
                )
                
                pr_data_list.append(pr_data)
            
            print(f"Retrieved {len(pr_data_list)} PRs for agent {agent}")
            return pr_data_list
    
    def mock_evaluate_pr_sequential(self, pr_data: PRData, model: str) -> EvaluationResult:
        """Mock evaluation that simulates API call delay and returns random scores - SEQUENTIAL."""
        print(f"    Evaluating PR {pr_data.pr_id} with {model}...")
        
        # Simulate API call delay (longer to mimic rate limit compliance)
        time.sleep(0.5)
        
        # Generate random scores between 1-5
        scores = {
            'coverage_purpose': random.randint(1, 5),
            'coverage_changes': random.randint(1, 5),
            'rationale_clarity': random.randint(1, 5),
            'commit_coverage': random.randint(1, 5),
            'testing_guidance': random.randint(1, 5),
            'readability': random.randint(1, 5)
        }
        
        return EvaluationResult(
            pr_id=pr_data.pr_id,
            agent=pr_data.agent,
            model=model,
            **scores
        )
    
    def evaluate_agent_sequential(self, agent: str, pr_data_list: List[PRData]):
        """Evaluate an agent with all models sequentially."""
        print(f"\n=== Evaluating agent: {agent} (Sequential Processing) ===")
        
        if not pr_data_list:
            print(f"No PRs found for agent {agent}")
            return {}
        
        agent_results = {}
        
        # Evaluate with each model sequentially
        for model in self.models:
            print(f"\n  Model: {model}")
            model_results = []
            
            # Process each PR sequentially
            for i, pr_data in enumerate(pr_data_list, 1):
                print(f"  PR {i}/{len(pr_data_list)}: {pr_data.pr_id}")
                result = self.mock_evaluate_pr_sequential(pr_data, model)
                model_results.append(result)
            
            # Calculate averages for this model
            if model_results:
                avg_scores = {}
                score_fields = ['coverage_purpose', 'coverage_changes', 'rationale_clarity', 
                               'commit_coverage', 'testing_guidance', 'readability']
                
                for field in score_fields:
                    scores = [getattr(r, field) for r in model_results]
                    avg_scores[field] = sum(scores) / len(scores)
                
                agent_results[model] = {
                    'total_evaluations': len(model_results),
                    'average_scores': avg_scores,
                    'overall_average': sum(avg_scores.values()) / len(avg_scores),
                    'results': [r.to_dict() for r in model_results],
                    'pr_ids_evaluated': sorted(list(set(r.pr_id for r in model_results)))
                }
                
                print(f"  {model}: {len(model_results)} evaluations, avg score: {agent_results[model]['overall_average']:.2f}")
        
        return agent_results
    
    def run_test_sequential(self, pr_count: int = 5):
        """Run a test evaluation with sequential processing."""
        print("Test Sequential PR Description Evaluation (Rate Limit Safe)")
        print("=" * 65)
        
        if not self.connect_database():
            return
        
        # Get all agents
        agents = self.get_agent_tables()
        print(f"Found {len(agents)} agents: {', '.join(agents)}")
        print(f"Using {len(self.models)} models: {', '.join(self.models)}")
        print(f"Sample size per agent: {pr_count} PRs")
        
        # First, get the same random PR samples for each agent
        print("\nSelecting random PR samples for each agent...")
        agent_pr_samples = {}
        for agent in agents[:1]:  # Test with first agent only
            pr_data_list = self.get_random_prs_for_agent(agent, pr_count)
            agent_pr_samples[agent] = pr_data_list
            print(f"  {agent}: {len(pr_data_list)} PRs selected")
        
        start_time = time.time()
        
        # Evaluate each agent with sequential processing
        all_results = {}
        for agent, pr_data_list in agent_pr_samples.items():
            agent_results = self.evaluate_agent_sequential(agent, pr_data_list)
            all_results[agent] = agent_results
        
        end_time = time.time()
        
        # Save test results
        test_results = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'execution_time_seconds': round(end_time - start_time, 2),
            'sample_size_per_agent': pr_count,
            'models_tested': self.models,
            'agents_tested': list(all_results.keys()),
            'processing_type': 'Sequential (Rate Limit Safe)',
            'results': all_results
        }
        
        output_file = "test_sequential_results.json"
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n{'='*65}")
        print("TEST COMPLETE")
        print(f"{'='*65}")
        print(f"Total execution time: {test_results['execution_time_seconds']} seconds")
        print(f"Processing type: Sequential (API rate limit safe)")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        total_evaluations = 0
        for agent, agent_data in all_results.items():
            print(f"\n{agent} Summary:")
            for model, model_data in agent_data.items():
                evals = model_data['total_evaluations']
                total_evaluations += evals
                print(f"  {model}: {model_data['overall_average']:.2f} ({evals} evaluations)")
        
        print(f"\nTotal evaluations completed: {total_evaluations}")
        print(f"Average time per evaluation: {test_results['execution_time_seconds']/total_evaluations:.2f} seconds")
        
        return test_results
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    """Main test function."""
    evaluator = TestSequentialEvaluator()
    
    try:
        evaluator.run_test_sequential(pr_count=5)  # Small test with 5 PRs
    except Exception as e:
        print(f"Error during test: {e}")
        raise
    finally:
        evaluator.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()