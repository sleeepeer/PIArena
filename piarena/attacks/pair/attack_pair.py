"""
PAIR (Prompt Automatic Iterative Refinement) attack for PIArena.

Adapts the PAIR jailbreaking method from "Jailbreaking Black Box Large Language
Models in Twenty Queries" (Chao et al., arXiv 2310.08419) to the PIArena prompt
injection benchmark.

PAIR uses an attacker LLM to iteratively refine a candidate injection prompt:
each round the attacker outputs a new prompt P; we query the target (through a
defense) with P, judge success; if success return P, else append (P, R, S) to
conversation and repeat.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Dict, Any, Optional, Callable, Tuple

from ..base import BaseAttack, register_attack
from ..utils import parse_attacker_feedback, refinement_judge
from .prompts import get_pair_attacker_system_prompt, format_pair_user_message
from ...utils import inject


# ---------------------------------------------------------------------------
# PAIR single-stream attack for one PIArena sample
# ---------------------------------------------------------------------------

def run_pair_for_sample(
    *,
    goal: str,
    context_tail: str,
    user_task: str,
    attacker_chat: Callable[[List[Dict[str, str]]], str],
    target_query: Callable[[str], str],
    judge_success: Callable[[str], bool],
    max_iterations: int = 10,
    verbose: bool = False,
) -> Tuple[Optional[str], bool, int, str]:
    """
    Run one PAIR stream for a single PIArena sample.

    Args:
        goal: Injected task (what we want the model to do).
        context_tail: Reference context excerpt (for attacker prompt).
        user_task: User's visible question (target_inst).
        attacker_chat: Function that takes messages and returns assistant content.
        target_query: Function that takes attack_prompt and returns target response.
        judge_success: Function that takes target response and returns True if
            it follows the goal.
        max_iterations: Maximum refinement rounds (paper uses K=3-12 per stream).
        verbose: Print progress.

    Returns:
        (best_attack_prompt, success, num_queries, best_response).
    """
    system_prompt = get_pair_attacker_system_prompt(
        goal, context_tail=context_tail, user_task=user_task,
    )
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    last_prompt: Optional[str] = None
    last_response: str = ""
    num_queries = 0
    success = False

    for iteration in range(max_iterations):
        # User message: first turn has no prior response/score
        if iteration == 0:
            user_content = format_pair_user_message(goal, None, None)
        else:
            user_content = format_pair_user_message(
                goal, last_response, 10 if success else 0,
            )
        conversation.append({"role": "user", "content": user_content})

        # Attacker generates P (and improvement)
        reply = attacker_chat(conversation)
        feedback = parse_attacker_feedback(reply)
        if feedback is None:
            print(f"  [PAIR] iter {iteration + 1}/{max_iterations}: invalid JSON from attacker, stopping")
            break
        last_prompt = feedback.prompt
        if not (last_prompt or "").strip():
            print(f"  [PAIR] iter {iteration + 1}/{max_iterations}: empty prompt, stopping")
            break

        # Query target (inject + defense + LLM) and judge
        last_response = target_query(last_prompt)
        num_queries += 1
        success = judge_success(last_response)
        score = 10 if success else 0

        # Always log iteration progress
        prompt_preview = last_prompt[:120].replace('\n', ' ')
        response_preview = last_response[:200].replace('\n', ' ')
        print(f"  [PAIR] iter {iteration + 1}/{max_iterations}: score={score}, "
              f"queries={num_queries}, prompt=\"{prompt_preview}...\"")
        print(f"    Response: \"{response_preview}{'...' if len(last_response) > 200 else ''}\"")

        if verbose:
            print(f"    Improvement: {feedback.improvement[:150]}{'...' if len(feedback.improvement) > 150 else ''}")

        if success:
            print(f"  [PAIR] SUCCESS at iter {iteration + 1}! queries={num_queries}")
            return (last_prompt, True, num_queries, last_response)

        # Append assistant reply for next iteration
        conversation.append({
            "role": "assistant",
            "content": json.dumps({"improvement": feedback.improvement, "prompt": feedback.prompt}),
        })

    return (last_prompt, False, num_queries, last_response)


# ---------------------------------------------------------------------------
# PAIR multi-stream wrapper
# ---------------------------------------------------------------------------

def run_pair_multistream(
    *,
    goal: str,
    context_tail: str,
    user_task: str,
    attacker_chat: Callable[[List[Dict[str, str]]], str],
    target_query: Callable[[str], str],
    judge_success: Callable[[str], bool],
    num_streams: int = 1,
    max_iterations_per_stream: int = 10,
    verbose: bool = False,
    parallel_streams: bool = True,
) -> Tuple[Optional[str], bool, int, str]:
    """
    Run multiple PAIR streams (paper: N streams, K iterations each).

    Returns as soon as one stream succeeds, or the best prompt across streams.
    When parallel_streams=True and num_streams>1, runs streams in parallel.

    Returns:
        (best_prompt, success, total_queries, best_response).
    """
    if num_streams <= 1:
        return run_pair_for_sample(
            goal=goal, context_tail=context_tail, user_task=user_task,
            attacker_chat=attacker_chat, target_query=target_query,
            judge_success=judge_success,
            max_iterations=max_iterations_per_stream, verbose=verbose,
        )

    if not parallel_streams:
        best_prompt: Optional[str] = None
        best_response: str = ""
        total_queries = 0
        for stream_idx in range(num_streams):
            if verbose:
                print(f"  ---- PAIR stream {stream_idx + 1}/{num_streams} ----")
            prompt, success, nq, response = run_pair_for_sample(
                goal=goal, context_tail=context_tail, user_task=user_task,
                attacker_chat=attacker_chat, target_query=target_query,
                judge_success=judge_success,
                max_iterations=max_iterations_per_stream, verbose=verbose,
            )
            total_queries += nq
            if success:
                return (prompt, True, total_queries, response)
            if prompt is not None and best_prompt is None:
                best_prompt = prompt
                best_response = response
        return (best_prompt, False, total_queries, best_response)

    # Parallel streams: run all in parallel, return first success or best
    best_prompt: Optional[str] = None
    best_response: str = ""
    total_queries = 0
    first_success: Optional[Tuple[Optional[str], int, str]] = None

    def run_one_stream(_stream_idx: int) -> Tuple[Optional[str], bool, int, str]:
        return run_pair_for_sample(
            goal=goal, context_tail=context_tail, user_task=user_task,
            attacker_chat=attacker_chat, target_query=target_query,
            judge_success=judge_success,
            max_iterations=max_iterations_per_stream, verbose=False,
        )

    with ThreadPoolExecutor(max_workers=num_streams) as ex:
        futures: List[Future] = [ex.submit(run_one_stream, i) for i in range(num_streams)]
        for f in as_completed(futures):
            prompt, success, nq, response = f.result()
            total_queries += nq
            if success and first_success is None:
                first_success = (prompt, total_queries, response)
                for ot in futures:
                    ot.cancel()
                break
            if prompt is not None and best_prompt is None:
                best_prompt = prompt
                best_response = response

    if first_success is not None:
        return (first_success[0], True, first_success[1], first_success[2])
    return (best_prompt, False, total_queries, best_response)


# ---------------------------------------------------------------------------
# PAIRAttack: BaseAttack interface for PIArena registry
# ---------------------------------------------------------------------------

@register_attack
class PAIRAttack(BaseAttack):
    """
    PAIR attack adapted for PIArena prompt injection.

    Uses an attacker LLM to iteratively refine injection prompts, evaluating
    each candidate through a defense before judging success.
    """

    name = "pair"
    DEFAULT_CONFIG = {
        "n_streams": 1,
        "n_iterations": 10,
        "temperature": 1.0,
        "max_tokens": 1024,
        "inject_position": "end",
        "pre_test": True,
        "verbose": False,
    }

    def execute(
        self,
        context,
        injected_task,
        target_inst="",
        target_task_answer="",
        injected_task_answer="",
        llm=None,
        attacker_llm=None,
        defense=None,
        inject_position=None,
        **kwargs,
    ):
        """
        Run PAIR attack for one sample.

        Args:
            context: Original document context.
            injected_task: The task we want to inject.
            target_inst: User's visible question.
            llm: Target LLM (Model instance).
            attacker_llm: Attacker LLM (Model instance). Falls back to llm.
            defense: Defense instance to evaluate against.
            inject_position: Where to inject ("end", "start", "random").

        Returns:
            Injected context string (context with best attack prompt embedded).
        """
        if attacker_llm is None:
            attacker_llm = llm
        if inject_position is None:
            inject_position = self.config.get("inject_position", "end")
        verbose = self.config.get("verbose", False)

        # Context tail for attacker prompt grounding
        context_tail = context[-1000:] if len(context) > 1000 else context

        # --- Callbacks ---
        def attacker_chat(messages: List[Dict[str, str]]) -> str:
            return attacker_llm.query(
                messages,
                max_new_tokens=self.config.get("max_tokens", 1024),
                temperature=self.config.get("temperature", 0.7),
                do_sample=True,
                top_p=0.95,
            )

        def target_query(attack_prompt: str) -> str:
            injected_context = inject(context, attack_prompt, inject_position=inject_position)
            result = defense.get_response(
                target_inst=target_inst,
                context=injected_context,
                llm=llm,
            )
            return result.get("response", "")

        def evaluator_chat(messages: List[Dict[str, str]]) -> str:
            return attacker_llm.query(
                messages,
                max_new_tokens=128,
                temperature=0.0,
            )

        def judge_success(response: str) -> bool:
            return refinement_judge(
                response=response,
                task_prompt=injected_task,
                evaluator_chat=evaluator_chat,
            )

        # --- Pre-test: try injected_task directly before refinement ---
        if self.config.get("pre_test", True):
            print("[PAIR] Pre-test: trying direct injection before refinement...")
            pre_response = target_query(injected_task)
            if judge_success(pre_response):
                print("[PAIR] Pre-test SUCCESS: direct injection already works, skipping refinement")
                return inject(context, injected_task, inject_position=inject_position)
            print("[PAIR] Pre-test failed, starting refinement")

        # --- Run PAIR ---
        n_streams = self.config.get("n_streams", 1)
        n_iterations = self.config.get("n_iterations", 10)

        best_prompt, success, num_queries, best_response = run_pair_multistream(
            goal=injected_task,
            context_tail=context_tail,
            user_task=target_inst,
            attacker_chat=attacker_chat,
            target_query=target_query,
            judge_success=judge_success,
            num_streams=n_streams,
            max_iterations_per_stream=n_iterations,
            verbose=verbose,
            parallel_streams=(n_streams > 1),
        )

        status = "SUCCESS" if success else "FAILED"
        print(f"[PAIR] {status}: queries={num_queries}, streams={n_streams}")
        if best_prompt:
            print(f"[PAIR] Final prompt:\n{best_prompt}")
            return inject(context, best_prompt, inject_position=inject_position)
        print("[PAIR] No valid prompt generated, returning original context")
        return context
