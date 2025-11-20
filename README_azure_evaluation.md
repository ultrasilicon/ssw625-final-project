# Multi-Model Azure OpenAI PR Description Evaluation

This tool evaluates AI-generated pull request descriptions using multiple Azure OpenAI models based on 6 criteria:

1. **Coverage of Purpose**: Does the description clearly explain why this PR was created?
2. **Coverage of Changes**: Does it describe all functional and structural changes introduced?
3. **Rationale Clarity**: Does it explain why the changes were made in this particular way?
4. **Commit Coverage**: Are all commits and their modifications adequately reflected in the description?
5. **Testing Guidance**: Does it provide instructions or information on how to test or validate the changes?
6. **Readability**: Is the description fluent, coherent, and concise?

## Available Models

- **Deepseek-V3-0324**: Latest DeepSeek model
- **DeepSeek-V3.1**: DeepSeek v3.1 
- **Grok-3**: Grok-3 model
- **GPT-4.1**: GPT-4.1 preview

## Setup

1. **Install dependencies:**
   ```bash
   pip install openai pandas
   ```

2. **Models are pre-configured** with Azure endpoints and API keys in the script.

## Usage

### Quick Test (2 PRs, 1 model, 1 agent)
```bash
python test_azure_evaluation.py
```

### Full Evaluation (20 PRs per agent, all models)
```bash
python evaluate_pr_descriptions_azure.py
```

## Output Structure

Results are saved in JSON format:

```
results/
├── Deepseek-V3-0324/
│   ├── Claude_Code_output.json
│   ├── Copilot_output.json
│   ├── Cursor_output.json
│   ├── Devin_output.json
│   ├── OpenAI_Codex_output.json
│   └── summary.json
├── DeepSeek-V3.1/
│   ├── [same structure]
├── Grok-3/
│   ├── [same structure]
└── GPT-4.1/
    ├── [same structure]
```

### Output Format

**Individual Agent Results** (`{agent}_output.json`):
```json
{
  "model": "Deepseek-V3-0324",
  "agent": "Copilot",
  "evaluation_date": "2025-11-19 15:30:45",
  "total_prs": 20,
  "results": [
    {
      "pr_id": 3214078227,
      "agent": "Copilot",
      "model": "Deepseek-V3-0324",
      "coverage_purpose": 4,
      "coverage_changes": 3,
      "rationale_clarity": 5,
      "commit_coverage": 2,
      "testing_guidance": 3,
      "readability": 4
    }
  ]
}
```

**Model Summary** (`summary.json`):
```json
{
  "model": "Deepseek-V3-0324",
  "evaluation_date": "2025-11-19 15:30:45",
  "agents": {
    "Copilot": {
      "total_prs": 20,
      "average_scores": {
        "coverage_purpose": 3.15,
        "coverage_changes": 3.85,
        "rationale_clarity": 2.90,
        "commit_coverage": 3.20,
        "testing_guidance": 2.75,
        "readability": 3.45
      },
      "overall_average": 3.22
    }
  }
}
```

## Console Output

The tool provides real-time console output showing:
- Progress for each model
- Progress for each agent
- Summary tables per model
- Final completion status

Example console output:
```
EVALUATION SUMMARY - Deepseek-V3-0324
================================================================================
Agent           | Coverage Purpose | Coverage Changes | Rationale Clarity | ... | Overall
Claude_Code     |             2.95 |             3.35 |              3.20 | ... |    3.12
Copilot         |             3.15 |             3.85 |              2.90 | ... |    3.22
```

## Database Requirements

- Requires `agent_pr.db` with agent-prefixed tables:
  - `{Agent}_pull_requests`
  - `{Agent}_commit_details`

## Scoring Scale

- **1**: Poor - Completely fails to meet the criterion
- **2**: Below Average - Barely addresses the criterion  
- **3**: Average - Adequately meets the criterion
- **4**: Good - Well addresses the criterion
- **5**: Excellent - Exceptionally well addresses the criterion

## Features

- ✅ **Multi-model support**: Evaluates with 4 different Azure models
- ✅ **JSON output**: Structured data for analysis
- ✅ **Progress tracking**: Real-time progress updates
- ✅ **Error handling**: Robust error handling and retries
- ✅ **Rate limiting**: Built-in delays to respect API limits
- ✅ **Reproducible**: Fixed random seed for consistent sampling