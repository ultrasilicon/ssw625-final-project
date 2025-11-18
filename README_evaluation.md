# PR Description Evaluation Tool

This tool evaluates AI-generated pull request descriptions using OpenAI GPT-4o-mini based on 6 criteria:

1. **Coverage of Purpose**: Does the description clearly explain why this PR was created?
2. **Coverage of Changes**: Does it describe all functional and structural changes introduced?
3. **Rationale Clarity**: Does it explain why the changes were made in this particular way?
4. **Commit Coverage**: Are all commits and their modifications adequately reflected in the description?
5. **Testing Guidance**: Does it provide instructions or information on how to test or validate the changes?
6. **Readability**: Is the description fluent, coherent, and concise?

## Setup

1. **Install dependencies:**
   ```bash
   pip install openai pandas
   ```

2. **Set up OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   Or add to your `.bashrc`/`.zshrc`:
   ```bash
   echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

## Usage

### Quick Test (2 PRs per agent)
```bash
python test_evaluation.py
```

### Full Evaluation (20 PRs per agent)
```bash
python evaluate_pr_descriptions.py
```

## Output

The tool generates:

1. **Individual agent result files**: `{Agent}_evaluation_results.txt`
   - Detailed scores for each PR
   - Average scores per criteria

2. **Console summary**: Comparison table across all agents

3. **Example output format**:
   ```
   Agent           | Coverage Purpose | Coverage Changes | Rationale Clarity | ...
   Claude_Code     |             2.95 |             3.35 |              3.20 | ...
   Copilot         |             3.12 |             3.87 |              2.94 | ...
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

## Notes

- Uses random scores if no OpenAI API key is provided (for testing)
- Processes 20 random PRs per agent by default
- Includes rate limiting delays to avoid API limits
- Limits file changes and commit data to avoid token limits