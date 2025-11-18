#!/usr/bin/env python3
"""
Script to evaluate PR descriptions using OpenAI GPT-4o-mini API
based on 6 criteria: coverage, changes, rationale, commits, testing, readability
"""

import sqlite3
import openai
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
import re
from dataclasses import dataclass
from enum import Enum

class EvaluationCriteria(Enum):
    COVERAGE_PURPOSE = "coverage_purpose"
    COVERAGE_CHANGES = "coverage_changes" 
    RATIONALE_CLARITY = "rationale_clarity"
    COMMIT_COVERAGE = "commit_coverage"
    TESTING_GUIDANCE = "testing_guidance"
    READABILITY = "readability"

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
    coverage_purpose: int
    coverage_changes: int
    rationale_clarity: int
    commit_coverage: int
    testing_guidance: int
    readability: int
    
class PRDescriptionEvaluator:
    def __init__(self, db_path="agent_pr.db", api_key=None):
        self.db_path = db_path
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        self.conn = None
        
        # Mapping of criteria to required data columns
        self.criteria_data_mapping = {
            EvaluationCriteria.COVERAGE_PURPOSE: ["title", "body"],
            EvaluationCriteria.COVERAGE_CHANGES: ["title", "body", "file_changes"],
            EvaluationCriteria.RATIONALE_CLARITY: ["title", "body"],
            EvaluationCriteria.COMMIT_COVERAGE: ["title", "body", "commit_messages", "file_changes"],
            EvaluationCriteria.TESTING_GUIDANCE: ["title", "body"],
            EvaluationCriteria.READABILITY: ["title", "body"]
        }
        
    def connect_database(self):
        """Connect to the unified database."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database {self.db_path} not found!")
        
        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")
    
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
    
    def get_random_prs_for_agent(self, agent: str, count: int = 20) -> List[PRData]:
        """Get random PRs for a specific agent with their commit details."""
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
                
            # Get commit details
            cursor.execute(f"""
                SELECT sha, message, filename, status, additions, deletions, patch
                FROM {commit_table}
                WHERE pr_id = ?
            """, (pr_id,))
            commit_rows = cursor.fetchall()
            
            commits = []
            commit_messages = []
            file_changes = []
            
            for commit_row in commit_rows:
                sha, message, filename, status, additions, deletions, patch = commit_row
                
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
                    'deletions': deletions,
                    'patch': patch[:500] if patch else ""  # Limit patch size
                })
            
            pr_data = PRData(
                pr_id=pr_row[0],
                title=pr_row[1] or "",
                body=pr_row[2] or "",
                agent=pr_row[3],
                commits=commits,
                commit_messages=list(set(commit_messages)),  # Remove duplicates
                file_changes=file_changes[:10]  # Limit to first 10 files to avoid token limits
            )
            
            pr_data_list.append(pr_data)
        
        print(f"Retrieved {len(pr_data_list)} PRs for agent {agent}")
        return pr_data_list
    
    def format_pr_data_for_evaluation(self, pr_data: PRData, criteria: EvaluationCriteria) -> str:
        """Format PR data based on the specific criteria being evaluated."""
        required_fields = self.criteria_data_mapping[criteria]
        
        formatted_data = f"PR ID: {pr_data.pr_id}\n"
        formatted_data += f"Agent: {pr_data.agent}\n"
        formatted_data += f"Title: {pr_data.title}\n"
        
        if "body" in required_fields:
            formatted_data += f"Description:\n{pr_data.body}\n"
        
        if "commit_messages" in required_fields:
            formatted_data += f"\nCommit Messages ({len(pr_data.commit_messages)}):\n"
            for i, msg in enumerate(pr_data.commit_messages[:5], 1):  # Limit to 5 commits
                formatted_data += f"{i}. {msg}\n"
        
        if "file_changes" in required_fields:
            formatted_data += f"\nFile Changes ({len(pr_data.file_changes)}):\n"
            for i, change in enumerate(pr_data.file_changes[:5], 1):  # Limit to 5 files
                formatted_data += f"{i}. {change['filename']} ({change['status']}, +{change['additions']}/-{change['deletions']})\n"
                if change['patch']:
                    formatted_data += f"   Patch preview: {change['patch'][:200]}...\n"
        
        return formatted_data
    
    def get_evaluation_prompt(self, criteria: EvaluationCriteria, pr_data_text: str) -> str:
        """Generate evaluation prompt for specific criteria."""
        
        criteria_descriptions = {
            EvaluationCriteria.COVERAGE_PURPOSE: "Coverage of purpose: Does the description clearly explain why this PR was created?",
            EvaluationCriteria.COVERAGE_CHANGES: "Coverage of changes: Does it describe all functional and structural changes introduced?",
            EvaluationCriteria.RATIONALE_CLARITY: "Rationale clarity: Does it explain why the changes were made in this particular way?",
            EvaluationCriteria.COMMIT_COVERAGE: "Commit coverage: Are all commits and their modifications adequately reflected in the description?",
            EvaluationCriteria.TESTING_GUIDANCE: "Testing guidance: Does it provide instructions or information on how to test or validate the changes?",
            EvaluationCriteria.READABILITY: "Readability: Is the description fluent, coherent, and concise?"
        }
        
        prompt = f"""Given the following PR information and AI-generated description, please evaluate it on this specific criterion:

{criteria_descriptions[criteria]}

Provide a score from 1 (poor) to 5 (excellent) based on this criterion only.

Score meanings:
1 = Poor: Completely fails to meet the criterion
2 = Below Average: Barely addresses the criterion
3 = Average: Adequately meets the criterion
4 = Good: Well addresses the criterion
5 = Excellent: Exceptionally well addresses the criterion

PR Information:
{pr_data_text}

Please respond with ONLY a single number (1-5) representing your score for this criterion. Do not include any explanation or additional text."""

        return prompt
    
    def evaluate_pr_with_openai(self, pr_data: PRData, criteria: EvaluationCriteria) -> int:
        """Evaluate a single PR on a single criterion using OpenAI GPT-4o-mini API."""
        if not self.client:
            print("Warning: OpenAI client not initialized. Using random scores.")
            return random.randint(1, 5)
        
        pr_text = self.format_pr_data_for_evaluation(pr_data, criteria)
        prompt = self.get_evaluation_prompt(criteria, pr_text)
        
        try:
            response = self.client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            # Extract score from response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract a number from the response
            score_match = re.search(r'[1-5]', response_text)
            if score_match:
                score = int(score_match.group())
                return max(1, min(5, score))  # Ensure score is between 1-5
            else:
                print(f"Warning: Could not parse score from response: {response_text}")
                return 3  # Default score
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return 3  # Default score
    
    def evaluate_agent_prs(self, agent: str, pr_count: int = 20) -> List[EvaluationResult]:
        """Evaluate all PRs for a specific agent on all criteria."""
        print(f"\nEvaluating {agent} PRs...")
        
        prs = self.get_random_prs_for_agent(agent, pr_count)
        if not prs:
            print(f"No PRs found for agent {agent}")
            return []
        
        results = []
        
        for i, pr_data in enumerate(prs, 1):
            print(f"  Evaluating PR {i}/{len(prs)} (ID: {pr_data.pr_id})")
            
            scores = {}
            for criteria in EvaluationCriteria:
                score = self.evaluate_pr_with_openai(pr_data, criteria)
                scores[criteria.value] = score
                time.sleep(0.1)  # Small delay to avoid rate limiting
            
            result = EvaluationResult(
                pr_id=pr_data.pr_id,
                agent=agent,
                coverage_purpose=scores['coverage_purpose'],
                coverage_changes=scores['coverage_changes'],
                rationale_clarity=scores['rationale_clarity'],
                commit_coverage=scores['commit_coverage'],
                testing_guidance=scores['testing_guidance'],
                readability=scores['readability']
            )
            
            results.append(result)
        
        return results
    
    def save_agent_results(self, agent: str, results: List[EvaluationResult]):
        """Save evaluation results for an agent to a text file."""
        filename = f"{agent}_evaluation_results.txt"
        
        with open(filename, 'w') as f:
            f.write(f"PR Description Evaluation Results - {agent}\n")
            f.write("=" * 60 + "\n\n")
            
            for result in results:
                f.write(f"PR ID: {result.pr_id}\n")
                f.write(f"Coverage of Purpose: {result.coverage_purpose}/5\n")
                f.write(f"Coverage of Changes: {result.coverage_changes}/5\n")
                f.write(f"Rationale Clarity: {result.rationale_clarity}/5\n")
                f.write(f"Commit Coverage: {result.commit_coverage}/5\n")
                f.write(f"Testing Guidance: {result.testing_guidance}/5\n")
                f.write(f"Readability: {result.readability}/5\n")
                f.write("-" * 40 + "\n")
            
            # Calculate averages
            if results:
                avg_coverage_purpose = sum(r.coverage_purpose for r in results) / len(results)
                avg_coverage_changes = sum(r.coverage_changes for r in results) / len(results)
                avg_rationale_clarity = sum(r.rationale_clarity for r in results) / len(results)
                avg_commit_coverage = sum(r.commit_coverage for r in results) / len(results)
                avg_testing_guidance = sum(r.testing_guidance for r in results) / len(results)
                avg_readability = sum(r.readability for r in results) / len(results)
                
                f.write(f"\nAVERAGE SCORES:\n")
                f.write(f"Coverage of Purpose: {avg_coverage_purpose:.2f}\n")
                f.write(f"Coverage of Changes: {avg_coverage_changes:.2f}\n")
                f.write(f"Rationale Clarity: {avg_rationale_clarity:.2f}\n")
                f.write(f"Commit Coverage: {avg_commit_coverage:.2f}\n")
                f.write(f"Testing Guidance: {avg_testing_guidance:.2f}\n")
                f.write(f"Readability: {avg_readability:.2f}\n")
        
        print(f"Results saved to {filename}")
    
    def print_summary(self, all_results: Dict[str, List[EvaluationResult]]):
        """Print summary of all evaluation results."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        criteria_names = [
            "Coverage of Purpose",
            "Coverage of Changes", 
            "Rationale Clarity",
            "Commit Coverage",
            "Testing Guidance",
            "Readability"
        ]
        
        # Print header
        print(f"{'Agent':<15} | " + " | ".join(f"{name:<18}" for name in criteria_names))
        print("-" * (15 + 3 + 21 * len(criteria_names)))
        
        for agent, results in all_results.items():
            if not results:
                continue
                
            averages = [
                sum(r.coverage_purpose for r in results) / len(results),
                sum(r.coverage_changes for r in results) / len(results),
                sum(r.rationale_clarity for r in results) / len(results),
                sum(r.commit_coverage for r in results) / len(results),
                sum(r.testing_guidance for r in results) / len(results),
                sum(r.readability for r in results) / len(results)
            ]
            
            avg_str = " | ".join(f"{avg:>18.2f}" for avg in averages)
            print(f"{agent:<15} | {avg_str}")
        
        print("\nScale: 1 (Poor) - 5 (Excellent)")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    """Main execution function."""
    print("PR Description Evaluation Tool")
    print("=" * 50)
    
    # Check for OpenAI API key
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("Proceeding with random scores for testing...")
    
    evaluator = PRDescriptionEvaluator(api_key=api_key)
    
    try:
        # Connect to database
        evaluator.connect_database()
        
        # Get all agents
        agents = evaluator.get_agent_tables()
        print(f"Found {len(agents)} agents: {', '.join(agents)}")
        
        all_results = {}
        
        # Evaluate each agent
        for agent in agents:
            results = evaluator.evaluate_agent_prs(agent, pr_count=20)
            all_results[agent] = results
            
            # Save results to file
            evaluator.save_agent_results(agent, results)
        
        # Print summary
        evaluator.print_summary(all_results)
        
        print(f"\nEvaluation complete!")
        print(f"Individual agent results saved to *_evaluation_results.txt files")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        evaluator.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()