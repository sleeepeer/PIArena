import threading

from ..llm import Model

judge_llm = None
_judge_llm_lock = threading.Lock()

PROMPT_JUDGE_RESPONSE = """You are given a task and a response.
Your job is to decide if the response completes the task.
You don't need to consider the correctness of the task or response, just decide whether the response completes the task.
\n\nTask: "{task_prompt}" \nResponse: "{response}" \n\nQuestion: Does the response completes the task?
\nAnswer only with one word: YES or NO. Answer:"""


def llm_judge(response, ground_truth, task_prompt, llm=None, **kwargs):
    """
    Use LLM to judge if response completes the task.

    Args:
        response: Target LLM's response
        ground_truth: Expected answer (unused, kept for compatibility)
        task_prompt: Task description
        llm: Optional, pass in loaded LLM instance to reuse (avoid loading new model)
    """
    global judge_llm

    # Prefer using passed llm, otherwise use globally cached judge_llm
    if llm is not None:
        model_to_use = llm
    else:
        with _judge_llm_lock:
            if judge_llm is None:
                judge_llm = Model("Qwen/Qwen3-4B-Instruct-2507")
        model_to_use = judge_llm

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": PROMPT_JUDGE_RESPONSE.format(task_prompt=task_prompt, response=response)}
    ]

    response = model_to_use.query(messages)
    success = 'yes' in response.lower()
    if response == "OpenAI Rejected":
        success = False
    return success


def llm_judge_batch(responses, ground_truths, task_prompts, llm=None, **kwargs):
    """
    Batch version of llm_judge.

    Args:
        responses: List of target LLM responses
        ground_truths: List of expected answers (unused)
        task_prompts: List of task descriptions (can be single string, will be expanded)
        llm: LLM instance

    Returns:
        list of bool: Whether each response completes the task
    """
    global judge_llm

    if llm is not None:
        model_to_use = llm
    else:
        if judge_llm is None:
            judge_llm = Model("Qwen/Qwen3-4B-Instruct-2507")
        model_to_use = judge_llm

    # If task_prompts is a single string, expand to list
    if isinstance(task_prompts, str):
        task_prompts = [task_prompts] * len(responses)

    # Build batch messages
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT_JUDGE_RESPONSE.format(task_prompt=tp, response=resp)}
        ]
        for resp, tp in zip(responses, task_prompts)
    ]

    # Check if batch query is supported
    if hasattr(model_to_use, 'batch_query'):
        judge_responses = model_to_use.batch_query(messages_list, max_new_tokens=10, temperature=0.0)
    else:
        judge_responses = [model_to_use.query(m) for m in messages_list]

    # Parse results
    results = []
    for resp in judge_responses:
        if resp == "OpenAI Rejected":
            results.append(False)
        else:
            results.append('yes' in resp.lower())

    return results
