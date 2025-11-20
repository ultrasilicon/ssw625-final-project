#!/usr/bin/env python3
"""
Script to evaluate PR descriptions using multiple Azure OpenAI models
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
from dataclasses import dataclass, asdict
from enum import Enum
# import concurrent.futures  # Commented out - using sequential processing to avoid API rate limits
import threading
from functools import partial

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
    model: str
    coverage_purpose: int
    coverage_changes: int
    rationale_clarity: int
    commit_coverage: int
    testing_guidance: int
    readability: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ModelConfig:
    name: str
    api_base_url: str
    api_version: str
    api_key: str

class MultiModelPRDescriptionEvaluator:
    def __init__(self, db_path="agent_pr.db", models_config_file="models_config.json"):
        self.db_path = db_path
        self.conn = None
        self.models_config = self.load_models_config(models_config_file)
        self._lock = threading.Lock()  # Thread safety for database operations
        
        # Mapping of criteria to required data columns
        self.criteria_data_mapping = {
            EvaluationCriteria.COVERAGE_PURPOSE: ["title", "body"],
            EvaluationCriteria.COVERAGE_CHANGES: ["title", "body", "file_changes"],
            EvaluationCriteria.RATIONALE_CLARITY: ["title", "body"],
            EvaluationCriteria.COMMIT_COVERAGE: ["title", "body", "commit_messages", "file_changes"],
            EvaluationCriteria.TESTING_GUIDANCE: ["title", "body"],
            EvaluationCriteria.READABILITY: ["title", "body"]
        }
        
    def load_models_config(self, config_file: str) -> Dict[str, ModelConfig]:
        """Load models configuration from JSON file."""
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration from your provided data
            config = {
                "models": {
                    "Deepseek-V3-0324": {
                        "api_base_url": "https://aider3.services.ai.azure.com",
                        "api_version": "2024-05-01-preview",
                        "api_key": "655HxlDivznudbfNrHtGAn5Z0KMlyD6hDsJ18Jsi7Xo25pvpS6EgJQQJ99BIACYeBjFXJ3w3AAAAACOGqvNF"
                    },
                    "DeepSeek-V3.1": {
                        "api_base_url": "https://aider3.services.ai.azure.com",
                        "api_version": "2024-05-01-preview",
                        "api_key": "655HxlDivznudbfNrHtGAn5Z0KMlyD6hDsJ18Jsi7Xo25pvpS6EgJQQJ99BIACYeBjFXJ3w3AAAAACOGqvNF"
                    },
                    "Grok-3": {
                        "api_base_url": "https://aider3.services.ai.azure.com",
                        "api_version": "2024-05-01-preview",
                        "api_key": "655HxlDivznudbfNrHtGAn5Z0KMlyD6hDsJ18Jsi7Xo25pvpS6EgJQQJ99BIACYeBjFXJ3w3AAAAACOGqvNF"
                    },
                    "GPT-4.1": {
                        "api_base_url": "https://aider3.cognitiveservices.azure.com",
                        "api_version": "2025-01-01-preview",
                        "api_key": "655HxlDivznudbfNrHtGAn5Z0KMlyD6hDsJ18Jsi7Xo25pvpS6EgJQQJ99BIACYeBjFXJ3w3AAAAACOGqvNF"
                    }
                }
            }
        
        models = {}
        for model_name, config_data in config["models"].items():
            models[model_name] = ModelConfig(
                name=model_name,
                api_base_url=config_data["api_base_url"],
                api_version=config_data["api_version"],
                api_key=config_data["api_key"]
            )
        
        return models
    
    def create_client_for_model(self, model_config: ModelConfig) -> openai.AzureOpenAI:
        """Create Azure OpenAI client for a specific model."""
        return openai.AzureOpenAI(
            api_key=model_config.api_key,
            api_version=model_config.api_version,
            azure_endpoint=model_config.api_base_url
        )
    
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
    
    def get_random_prs_for_agent(self, agent: str, count: int = 200) -> List[PRData]:
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
    
    def evaluate_pr_with_model(self, pr_data: PRData, criteria: EvaluationCriteria, model_name: str, client: openai.AzureOpenAI) -> int:
        """Evaluate a single PR on a single criterion using Azure OpenAI model."""
        pr_text = self.format_pr_data_for_evaluation(pr_data, criteria)
        prompt = self.get_evaluation_prompt(criteria, pr_text)
        
        try:
            response = client.chat.completions.create(
                model=model_name,
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
            print(f"Error calling Azure OpenAI API for model {model_name}: {e}")
            return 3  # Default score
    
    def evaluate_agent_prs_with_model(self, agent: str, model_name: str, pr_count: int = 200) -> List[EvaluationResult]:
        """Evaluate all PRs for a specific agent using a specific model on all criteria."""
        print(f"\nEvaluating {agent} PRs with model {model_name}...")
        
        model_config = self.models_config.get(model_name)
        if not model_config:
            print(f"Model {model_name} not found in configuration!")
            return []
        
        client = self.create_client_for_model(model_config)
        prs = self.get_random_prs_for_agent(agent, pr_count)
        
        if not prs:
            print(f"No PRs found for agent {agent}")
            return []
        
        results = []
        
        # Use sequential processing to avoid API rate limits
        for i, pr_data in enumerate(prs, 1):
            print(f"  Evaluating PR {i}/{len(prs)} (ID: {pr_data.pr_id})")
            
            scores = {}
            for criteria in EvaluationCriteria:
                score = self.evaluate_pr_with_model(pr_data, criteria, model_name, client)
                scores[criteria.value] = score
                time.sleep(0.5)  # Increased delay to avoid rate limiting
            
            result = EvaluationResult(
                pr_id=pr_data.pr_id,
                agent=agent,
                model=model_name,
                coverage_purpose=scores['coverage_purpose'],
                coverage_changes=scores['coverage_changes'],
                rationale_clarity=scores['rationale_clarity'],
                commit_coverage=scores['commit_coverage'],
                testing_guidance=scores['testing_guidance'],
                readability=scores['readability']
            )
            
            results.append(result)
        
        print(f"  Completed evaluation of {len(results)} PRs for {agent}")
        return results
    
    def evaluate_agent_prs_with_model_fixed_prs(self, agent: str, model_name: str, pr_data_list: List[PRData]) -> List[EvaluationResult]:
        """Evaluate a specific list of PRs for an agent using a specific model on all criteria."""
        print(f"\nEvaluating {agent} PRs with model {model_name} (using fixed PR set)...")
        
        model_config = self.models_config.get(model_name)
        if not model_config:
            print(f"Model {model_name} not found in configuration!")
            return []
        
        client = self.create_client_for_model(model_config)
        
        if not pr_data_list:
            print(f"No PRs provided for agent {agent}")
            return []
        
        results = []
        
        # Use sequential processing to avoid API rate limits
        for i, pr_data in enumerate(pr_data_list, 1):
            print(f"  Evaluating PR {i}/{len(pr_data_list)} (ID: {pr_data.pr_id})")
            
            scores = {}
            for criteria in EvaluationCriteria:
                score = self.evaluate_pr_with_model(pr_data, criteria, model_name, client)
                scores[criteria.value] = score
                time.sleep(0.5)  # Increased delay to avoid rate limiting
            
            result = EvaluationResult(
                pr_id=pr_data.pr_id,
                agent=agent,
                model=model_name,
                coverage_purpose=scores['coverage_purpose'],
                coverage_changes=scores['coverage_changes'],
                rationale_clarity=scores['rationale_clarity'],
                commit_coverage=scores['commit_coverage'],
                testing_guidance=scores['testing_guidance'],
                readability=scores['readability']
            )
            
            results.append(result)
        
        print(f"  Completed evaluation of {len(results)} PRs for {agent}")
        return results
    
    def save_results_json(self, model_name: str, agent: str, results: List[EvaluationResult]):
        """Save evaluation results to JSON format."""
        output_dir = Path("results") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        output_file = output_dir / f"{agent}_output.json"
        results_data = {
            "model": model_name,
            "agent": agent,
            "evaluation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_prs": len(results),
            "results": [result.to_dict() for result in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def save_summary_json(self, model_name: str, all_results: Dict[str, List[EvaluationResult]]):
        """Save summary of all evaluation results for a model to JSON."""
        output_dir = Path("results") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            "model": model_name,
            "evaluation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "agents": {}
        }
        
        for agent, results in all_results.items():
            if not results:
                continue
                
            averages = {
                "coverage_purpose": sum(r.coverage_purpose for r in results) / len(results),
                "coverage_changes": sum(r.coverage_changes for r in results) / len(results),
                "rationale_clarity": sum(r.rationale_clarity for r in results) / len(results),
                "commit_coverage": sum(r.commit_coverage for r in results) / len(results),
                "testing_guidance": sum(r.testing_guidance for r in results) / len(results),
                "readability": sum(r.readability for r in results) / len(results)
            }
            
            summary_data["agents"][agent] = {
                "total_prs": len(results),
                "average_scores": averages,
                "overall_average": sum(averages.values()) / len(averages)
            }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Summary saved to {summary_file}")
    
    def print_model_summary(self, model_name: str, all_results: Dict[str, List[EvaluationResult]]):
        """Print summary of evaluation results for a specific model."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY - {model_name}")
        print(f"{'='*80}")
        
        criteria_names = [
            "Coverage Purpose",
            "Coverage Changes", 
            "Rationale Clarity",
            "Commit Coverage",
            "Testing Guidance",
            "Readability"
        ]
        
        # Print header
        print(f"{'Agent':<15} | " + " | ".join(f"{name:<15}" for name in criteria_names) + " | Overall")
        print("-" * (15 + 3 + 18 * len(criteria_names) + 10))
        
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
            
            overall_avg = sum(averages) / len(averages)
            avg_str = " | ".join(f"{avg:>15.2f}" for avg in averages)
            print(f"{agent:<15} | {avg_str} | {overall_avg:>7.2f}")
        
        print("\nScale: 1 (Poor) - 5 (Excellent)")
    
    def run_evaluation(self, pr_count: int = 200):
        """Run evaluation across all models and agents with sequential processing to avoid API rate limits."""
        print("Multi-Model PR Description Evaluation Tool")
        print("=" * 60)
        
        # Connect to database
        self.connect_database()
        
        # Get all agents
        agents = self.get_agent_tables()
        print(f"Found {len(agents)} agents: {', '.join(agents)}")
        print(f"Found {len(self.models_config)} models: {', '.join(self.models_config.keys())}")
        print(f"Sample size per agent: {pr_count} PRs")
        
        # First, get the same random PR samples for each agent (this ensures consistency across models)
        print("\nSelecting random PR samples for each agent...")
        agent_pr_samples = {}
        for agent in agents:
            pr_data_list = self.get_random_prs_for_agent(agent, pr_count)
            agent_pr_samples[agent] = pr_data_list
            print(f"  {agent}: {len(pr_data_list)} PRs selected")
        
        # Now evaluate each model using the SAME PR samples
        for model_name in self.models_config.keys():
            print(f"\n{'='*60}")
            print(f"EVALUATING WITH MODEL: {model_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            # Evaluate each agent with this model using the pre-selected PRs
            for agent in agents:
                if agent in agent_pr_samples and agent_pr_samples[agent]:
                    results = self.evaluate_agent_prs_with_model_fixed_prs(
                        agent, model_name, agent_pr_samples[agent]
                    )
                    model_results[agent] = results
                    
                    # Save individual agent results
                    self.save_results_json(model_name, agent, results)
                else:
                    print(f"No PRs available for agent {agent}, skipping...")
            
            # Save model summary
            self.save_summary_json(model_name, model_results)
            
            # Print model summary
            self.print_model_summary(model_name, model_results)
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print("Results saved in results/ directory structure:")
        print("  results/[model]/[agent]_output.json")
        print("  results/[model]/summary.json")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    """Main execution function."""
    evaluator = MultiModelPRDescriptionEvaluator()
    
    try:
        evaluator.run_evaluation(pr_count=300)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    finally:
        evaluator.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()