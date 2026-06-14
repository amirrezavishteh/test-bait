# Update

To eliminate the need for tedious hyperparameter tuning of detection thresholds across different models, we have implemented an additional post-processing module. This module employs a secondary LLM to evaluate whether inverted target sequences contain potentially malicious content.

## Workflow
1. Generate a candidate attack target sequence
2. Instead of relying solely on Q-score for evaluation
3. Utilize a secondary LLM to analyze the sequence for malicious content

This approach provides a more robust and model-agnostic method for detecting potentially harmful content in inverted sequences.

## Configuration
The system is configured through two main components:
- System prompt for judging attack targets: `JUDGE_SYSTEM_PROMPT` in `src/utils/constants.py`
- Judge model selection: `judge_model_name` in `src/config/arguments.py`