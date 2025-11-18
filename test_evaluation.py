#!/usr/bin/env python3
"""
Test script to demonstrate PR evaluation with just a few samples per agent.
This can be run with or without an OpenAI API key for testing.
"""

from evaluate_pr_descriptions import PRDescriptionEvaluator
import os

def test_evaluation():
    """Test the evaluation system with small samples."""
    print("PR Description Evaluation - Test Run")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found - will use real GPT-4o-mini evaluations")
    else:
        print("ℹ️  No OpenAI API key - using random scores for testing")
    
    evaluator = PRDescriptionEvaluator(api_key=api_key)
    
    try:
        evaluator.connect_database()
        agents = evaluator.get_agent_tables()
        print(f"Found {len(agents)} agents: {', '.join(agents)}")
        
        # Test with just 2 PRs per agent for quick testing
        all_results = {}
        
        for agent in agents[:2]:  # Test only first 2 agents
            print(f"\nTesting {agent} with 2 PRs...")
            results = evaluator.evaluate_agent_prs(agent, pr_count=2)
            all_results[agent] = results
            
            if results:
                print(f"✅ Successfully evaluated {len(results)} PRs for {agent}")
                # Show sample scores
                result = results[0]
                print(f"   Sample scores: Purpose={result.coverage_purpose}, Changes={result.coverage_changes}, Readability={result.readability}")
        
        # Print mini summary
        if all_results:
            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)
            
            for agent, results in all_results.items():
                if results:
                    avg_purpose = sum(r.coverage_purpose for r in results) / len(results)
                    avg_changes = sum(r.coverage_changes for r in results) / len(results)
                    avg_readability = sum(r.readability for r in results) / len(results)
                    
                    print(f"{agent:<15}: Purpose={avg_purpose:.1f}, Changes={avg_changes:.1f}, Readability={avg_readability:.1f}")
        
        print("\n✅ Test completed successfully!")
        print("\nTo run full evaluation with 20 PRs per agent:")
        print("   python evaluate_pr_descriptions.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        evaluator.close()

if __name__ == "__main__":
    test_evaluation()