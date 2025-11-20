#!/usr/bin/env python3
"""
Test script for parallel evaluation logic without Azure OpenAI dependencies
"""

import sqlite3
import json
import random
import time
from pathlib import Path
from typing import Dict, List
import concurrent.futures
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

class TestParallelEvaluator:
    def __init__(self, db_path="agent_pr.db"):
        self.db_path = db_path
        self.conn = None
        self._lock = threading.Lock()
        
        # Mock models for testing
        self.models = ['Deepseek-V3-0324', 'DeepSeek-V3.1', 'Grok-3', 'GPT-4.1']
        
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
    
    def get_random_prs_for_agent(self, agent: str, count: int = 300) -> List[PRData]:
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
                    
                # Get commit details (simplified for testing)
                cursor.execute(f"""
                    SELECT sha, message, filename, status, additions, deletions
                    FROM {commit_table}
                    WHERE pr_id = ? LIMIT 5
                """, (pr_id,))
                commit_rows = cursor.fetchall()
                
                commits = []
                commit_messages = []
                file_changes = []
                
                for commit_row in commit_rows:
                    sha, message, filename, status, additions, deletions = commit_row
                    
                    if message and message not in commit_messages:
                        commit_messages.append(message)
                    
                    commits.append({
                        'sha': sha,
                        'message': message,
                        'filename': filename,
                        'status': status,
                        'additions': additions,
                        'deletions': deletions
                    })
                    
                    file_changes.append({
                        'filename': filename,
                        'status': status,
                        'additions': additions,
                        'deletions': deletions
                    })
                
                pr_data = PRData(
                    pr_id=pr_row[0],
                    title=pr_row[1] or "",
                    body=pr_row[2] or "",
                    agent=pr_row[3],
                    commits=commits,
                    commit_messages=list(set(commit_messages)),
                    file_changes=file_changes[:5]  # Limit to first 5 files
                )
                
                pr_data_list.append(pr_data)
            
            print(f"Retrieved {len(pr_data_list)} PRs for agent {agent}")
            return pr_data_list
    
    def mock_evaluate_pr(self, pr_data: PRData, model: str) -> EvaluationResult:
        """Mock evaluation that simulates API call delay and returns random scores."""
        # Simulate API call delay
        time.sleep(random.uniform(0.1, 0.3))
        
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
    
    def evaluate_agent_with_parallel_models(self, agent: str, pr_count: int = 10):
        """Evaluate an agent with all models in parallel."""
        print(f"\n=== Evaluating agent: {agent} ===")
        
        # Get PRs for this agent
        pr_data_list = self.get_random_prs_for_agent(agent, pr_count)
        
        if not pr_data_list:
            print(f"No PRs found for agent {agent}")
            return {}
        
        # Create all evaluation tasks (PR x Model combinations)
        evaluation_tasks = []
        for pr_data in pr_data_list:
            for model in self.models:
                evaluation_tasks.append((pr_data, model))
        
        print(f"Running {len(evaluation_tasks)} evaluations in parallel...")
        
        # Run evaluations in parallel
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.mock_evaluate_pr, pr_data, model): (pr_data, model)
                for pr_data, model in evaluation_tasks
            }
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                pr_data, model = future_to_task[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    completed += 1
                    if completed % 20 == 0:
                        print(f"  Completed {completed}/{len(evaluation_tasks)} evaluations...")
                except Exception as e:
                    print(f"  Error evaluating PR {pr_data.pr_id} with {model}: {e}")
        
        # Organize results by model
        model_results = {}
        for model in self.models:
            model_data = [r for r in all_results if r.model == model]
            
            if model_data:
                # Calculate averages
                avg_scores = {}
                score_fields = ['coverage_purpose', 'coverage_changes', 'rationale_clarity', 
                               'commit_coverage', 'testing_guidance', 'readability']
                
                for field in score_fields:
                    scores = [getattr(r, field) for r in model_data]
                    avg_scores[field] = sum(scores) / len(scores)
                
                model_results[model] = {
                    'total_evaluations': len(model_data),
                    'average_scores': avg_scores,
                    'overall_average': sum(avg_scores.values()) / len(avg_scores),
                    'results': [r.to_dict() for r in model_data]
                }
                
                print(f"  {model}: {len(model_data)} evaluations, avg score: {model_results[model]['overall_average']:.2f}")
        
        return model_results
    
    def run_test_evaluation(self, pr_count: int = 10):
        """Run a test evaluation with parallel processing using same PRs for all models."""
        print("Test Parallel PR Description Evaluation (Fixed PR Sets)")
        print("=" * 60)
        
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
        for agent in agents:
            pr_data_list = self.get_random_prs_for_agent(agent, pr_count)
            agent_pr_samples[agent] = pr_data_list
            print(f"  {agent}: {len(pr_data_list)} PRs selected")
        
        start_time = time.time()
        
        # Now evaluate each model using the SAME PR samples
        all_results = {}
        for agent, pr_data_list in agent_pr_samples.items():
            print(f"\n=== Evaluating agent: {agent} (using fixed PR set) ===")
            
            if not pr_data_list:
                print(f"No PRs found for agent {agent}")
                continue
            
            # Create all evaluation tasks (PR x Model combinations) using the SAME PRs
            evaluation_tasks = []
            for pr_data in pr_data_list:
                for model in self.models:
                    evaluation_tasks.append((pr_data, model))
            
            print(f"Running {len(evaluation_tasks)} evaluations in parallel (same PRs for all models)...")
            
            # Run evaluations in parallel
            agent_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.mock_evaluate_pr, pr_data, model): (pr_data, model)
                    for pr_data, model in evaluation_tasks
                }
                
                # Collect results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_task):
                    pr_data, model = future_to_task[future]
                    try:
                        result = future.result()
                        agent_results.append(result)
                        completed += 1
                        if completed % 20 == 0:
                            print(f"  Completed {completed}/{len(evaluation_tasks)} evaluations...")
                    except Exception as e:
                        print(f"  Error evaluating PR {pr_data.pr_id} with {model}: {e}")
            
            # Organize results by model
            model_results = {}
            for model in self.models:
                model_data = [r for r in agent_results if r.model == model]
                
                if model_data:
                    # Calculate averages
                    avg_scores = {}
                    score_fields = ['coverage_purpose', 'coverage_changes', 'rationale_clarity', 
                                   'commit_coverage', 'testing_guidance', 'readability']
                    
                    for field in score_fields:
                        scores = [getattr(r, field) for r in model_data]
                        avg_scores[field] = sum(scores) / len(scores)
                    
                    model_results[model] = {
                        'total_evaluations': len(model_data),
                        'average_scores': avg_scores,
                        'overall_average': sum(avg_scores.values()) / len(avg_scores),
                        'results': [r.to_dict() for r in model_data],
                        'pr_ids_evaluated': sorted(list(set(r.pr_id for r in model_data)))  # Show which PRs were evaluated
                    }
                    
                    print(f"  {model}: {len(model_data)} evaluations (PRs: {len(set(r.pr_id for r in model_data))}), avg score: {model_results[model]['overall_average']:.2f}")
            
            all_results[agent] = model_results
        
        end_time = time.time()
        
        # Verify that all models evaluated the same PRs for each agent
        print(f"\n{'='*60}")
        print("VERIFICATION: Same PRs across all models")
        print(f"{'='*60}")
        for agent, agent_data in all_results.items():
            print(f"\n{agent}:")
            pr_sets = {}
            for model, model_data in agent_data.items():
                pr_ids = set(model_data['pr_ids_evaluated'])
                pr_sets[model] = pr_ids
                print(f"  {model}: {len(pr_ids)} PRs")
            
            # Check if all models evaluated the same PRs
            all_pr_sets = list(pr_sets.values())
            if all_pr_sets and all(pr_set == all_pr_sets[0] for pr_set in all_pr_sets):
                print(f"  ✅ All models evaluated the SAME {len(all_pr_sets[0])} PRs")
            else:
                print(f"  ❌ Models evaluated different PR sets!")
        
        # Save test results
        test_results = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'execution_time_seconds': round(end_time - start_time, 2),
            'sample_size_per_agent': pr_count,
            'models_tested': self.models,
            'agents_tested': list(all_results.keys()),
            'verification': 'Same PRs used across all models for fair comparison',
            'results': all_results
        }
        
        output_file = "test_parallel_fixed_prs_results.json"
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TEST COMPLETE")
        print(f"{'='*60}")
        print(f"Total execution time: {test_results['execution_time_seconds']} seconds")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        for agent, agent_data in all_results.items():
            print(f"\n{agent} Summary:")
            for model, model_data in agent_data.items():
                print(f"  {model}: {model_data['overall_average']:.2f}")
        
        return test_results
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    """Main test function."""
    evaluator = TestParallelEvaluator()
    
    try:
        evaluator.run_test_evaluation(pr_count=300)  # Small test with 300 PRs per agent
    except Exception as e:
        print(f"Error during test: {e}")
        raise
    finally:
        evaluator.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()