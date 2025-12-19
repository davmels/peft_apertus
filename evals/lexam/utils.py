import ast
import logging
import re
import sys
import yaml
from typing import Any, Dict, List

MCQ_PROMPT_TEMPLATE = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., A, B, C, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., A, B, C, etc.) with a brief justification for why it best fits the legal analysis.

Format your final answer as follows:
 Correct Answer: ###C### 

Question:
 {question}

Answer:"""

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML config: {e}")

def setup_logger(rank: int = 0):
    logger = logging.getLogger(f"Rank{rank}")
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers if they exist
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def _safe_literal_eval_choices(x: Any):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v
        except Exception:
            return x
    return x

def get_choice_labels(num_choices: int) -> List[str]:
    # Generate A, B, C, ... Z, AA, AB, ...
    labels = []
    for i in range(num_choices):
        s = ""
        curr = i
        while True:
            s = chr(65 + (curr % 26)) + s
            curr = curr // 26 - 1
            if curr < 0:
                break
        labels.append(s)
    return labels

def format_prompt(sample: Dict[str, Any]) -> str:
    question_text = sample["question"]
    choice_list = _safe_literal_eval_choices(sample["choices"])

    if isinstance(choice_list, list):
        labels = get_choice_labels(len(choice_list))
        for label, text in zip(labels, choice_list):
            question_text += f"\n{label}. {text}"

    return MCQ_PROMPT_TEMPLATE.format(
        course_name=sample.get("course", "Law"),
        question=question_text,
    )

def parse_choice(generated_text: str, match_choice_regex: str) -> str:
    matches = re.findall(match_choice_regex, generated_text)
    return matches[-1] if matches else "None"
