from .longbench_metrics import (
    qa_f1_score,
    rouge_score,
    retrieval_score,
    code_sim_score,
)
from .llm_judge import (
    llm_judge,
    llm_judge_batch,
)
from .open_prompt_injection_metrics import (
    open_prompt_injection_utility,
)

longbench_metric_dict = {
    "qasper": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "gov_report": rouge_score,
    "multi_news": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
}

def substring_match(response, ground_truth, task_prompt, **kwargs):
    return ground_truth.lower() in response.lower()

def start_with_match(response, ground_truth, task_prompt, **kwargs):
    return response.lower().strip().startswith(ground_truth.lower().strip())

EVALUATIONS = {
    "substring": substring_match,
    "start_with": start_with_match,
    "longbench": longbench_metric_dict,
    "open_prompt_injection": open_prompt_injection_utility,
}
