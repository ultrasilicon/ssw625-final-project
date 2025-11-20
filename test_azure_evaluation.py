#!/usr/bin/env python3
"""
Test script to verify the Azure multi-model evaluation setup.
Tests with a small sample to ensure everything works.
"""

from evaluate_pr_descriptions_azure import MultiModelPRDescriptionEvaluator
import json

def test_azure_evaluation():
    """Test the Azure multi-model evaluation system."""
    print("Azure Multi-Model PR Description Evaluation - Test")
    print("=" * 60)
    
    evaluator = MultiModelPRDescriptionEvaluator()
    
    try:
        # Test with just one model and small samples
        evaluator.connect_database()
        agents = evaluator.get_agent_tables()
        
        print(f"Available agents: {', '.join(agents)}")
        print(f"Available models: {', '.join(evaluator.models_config.keys())}")
        
        # Test with first model and first agent only
        test_model = list(evaluator.models_config.keys())[0]
        test_agent = agents[0]
        
        print(f"\nTesting with model: {test_model}")
        print(f"Testing with agent: {test_agent}")
        
        # Test with just 2 PRs
        results = evaluator.evaluate_agent_prs_with_model(test_agent, test_model, pr_count=2)
        
        if results:
            print(f"✅ Successfully evaluated {len(results)} PRs")
            
            # Save test results
            evaluator.save_results_json(test_model, test_agent, results)
            
            # Show sample result
            result = results[0]
            print(f"\nSample result:")
            print(f"  PR ID: {result.pr_id}")
            print(f"  Agent: {result.agent}")
            print(f"  Model: {result.model}")
            print(f"  Scores: Purpose={result.coverage_purpose}, Changes={result.coverage_changes}")
            
            # Save test summary
            test_results = {test_agent: results}
            evaluator.save_summary_json(test_model, test_results)
            
            print(f"\n✅ Test completed successfully!")
            print(f"Check results/{test_model}/ for output files")
            
        else:
            print("❌ No results generated")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        evaluator.close()

if __name__ == "__main__":
    test_azure_evaluation()