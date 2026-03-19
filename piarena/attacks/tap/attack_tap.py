"""
Tree of Attacks with Pruning (TAP) attack for PIArena.

Adapts the TAP jailbreaking method from "Tree of Attacks: Jailbreaking Black-Box
LLMs Automatically" (Mehrotra et al., NeurIPS 2024) to the PIArena prompt
injection benchmark.

For each PIArena sample, TAP uses an attacker LLM to iteratively refine candidate
injection prompts in a tree: branch with the attacker, prune by on-topic check
and score (judge success), keep top-k nodes, repeat until success or max depth.
"""

from __future__ import annotations

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple

from ..base import BaseAttack, register_attack
from ..utils import parse_attacker_feedback, AttackerFeedback, refinement_judge
from .prompts import (
    get_tap_attacker_system_prompt,
    get_tap_on_topic_prompt,
    get_tap_scorer_prompt,
)
from ...utils import inject


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class TapNode:
    """Single node in the TAP tree."""
    conversation: List[Dict[str, str]]
    feedback: Optional[AttackerFeedback] = None
    response: Optional[str] = None
    on_topic: Optional[bool] = None
    score: Optional[int] = None
    success: bool = False
    children: List["TapNode"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TAP attack loop for one PIArena sample
# ---------------------------------------------------------------------------

def run_tap_for_sample(
    *,
    goal: str,
    context_tail: str,
    user_task: str,
    attacker_chat: Callable[[List[Dict[str, str]]], str],
    target_query: Callable[[str], str],
    judge_success: Callable[[str], bool],
    evaluator_chat: Callable[[List[Dict[str, str]]], str],
    root_nodes: int = 1,
    branching_factor: int = 3,
    width: int = 5,
    depth: int = 5,
    parallel_branching: bool = True,
) -> Tuple[Optional[str], bool, List[TapNode]]:
    """
    Run one TAP tree for a single PIArena sample.

    Args:
        goal: Injected task (what we want the model to do).
        context_tail: Reference context excerpt (where injection goes).
        user_task: User's visible question (target_inst).
        attacker_chat: Attacker LLM query function.
        target_query: Function: attack_prompt -> target model response
            (handles inject + defense + LLM internally).
        judge_success: Function: response -> bool (binary success, aligned
            with the final ASR evaluator).
        evaluator_chat: Evaluator LLM query function (for on-topic + scoring).
        root_nodes: Number of root nodes (parallel starting points).
        branching_factor: Children per node per iteration.
        width: Max nodes to keep after pruning.
        depth: Max tree depth (iterations).
        parallel_branching: Generate branches in parallel.

    Returns:
        (best_attack_prompt, success, final_nodes).
    """
    on_topic_sys = get_tap_on_topic_prompt(goal)
    scorer_sys = get_tap_scorer_prompt(goal)
    attacker_sys = get_tap_attacker_system_prompt(
        goal, context_tail=context_tail, user_task=user_task,
    )

    def _attacker_messages(node: TapNode) -> List[Dict[str, str]]:
        """Build attacker message list: system + conversation history."""
        msgs = [{"role": "system", "content": attacker_sys}]
        for m in node.conversation:
            if m["role"] != "system":
                msgs.append(m)
        return msgs

    def _check_on_topic(prompt: str) -> bool:
        """Check if the attack prompt is on-topic (aims to achieve the goal)."""
        if not (prompt or "").strip():
            return False
        out = evaluator_chat([
            {"role": "system", "content": on_topic_sys},
            {"role": "user", "content": f"Attack prompt (P):\n{prompt}"},
        ])
        return "yes" in (out or "").lower()

    def _score_response(prompt: str, response: str) -> int:
        """Score how well the response follows the goal (1-10)."""
        out = evaluator_chat([
            {"role": "system", "content": scorer_sys},
            {"role": "user", "content": f"Prompt (P): {prompt}\nResponse (R): {response}"},
        ])
        m = re.search(r"\d+", out or "")
        if m:
            return max(1, min(10, int(m.group(0))))
        return 0

    # Initialize root nodes
    roots: List[TapNode] = [TapNode(conversation=[]) for _ in range(root_nodes)]
    current_nodes: List[TapNode] = roots
    best_prompt: Optional[str] = None
    best_score: int = 0
    found_success: bool = False

    for iteration in range(depth):
        print(f"  [TAP] depth {iteration + 1}/{depth}, {len(current_nodes)} active nodes")
        next_nodes: List[TapNode] = []

        for node in current_nodes:
            # Build user message with previous response and score
            prev_response = node.response if node.response else "[First attempt]"
            prev_score = node.score if node.score is not None else "[First attempt]"
            user_content = (
                f"Goal (G): {goal}\n"
                f"Response (R): {prev_response}\n"
                f"Score (S): {prev_score}"
            )
            node.conversation.append({"role": "user", "content": user_content})

            def gen_one_branch(_: int, node=node) -> Optional[Tuple[TapNode, AttackerFeedback]]:
                """Generate one child branch: attacker -> on-topic check -> target query -> score."""
                reply = attacker_chat(_attacker_messages(node))
                feedback = parse_attacker_feedback(reply)
                if feedback is None:
                    return None

                # Build child node with extended conversation
                child_conv = list(node.conversation)
                child_conv.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "improvement": feedback.improvement,
                        "prompt": feedback.prompt,
                    }),
                })
                child = TapNode(conversation=child_conv, feedback=feedback)

                # Phase 1 prune: on-topic check
                if not _check_on_topic(feedback.prompt):
                    return None
                child.on_topic = True

                # Query target (inject + defense + LLM)
                child.response = target_query(feedback.prompt)

                # Check binary success (aligned with final ASR judge)
                child.success = judge_success(child.response)
                if child.success:
                    child.score = 10
                else:
                    child.score = _score_response(feedback.prompt, child.response or "")

                return (child, feedback)

            # Generate branches (parallel or sequential)
            if parallel_branching and branching_factor > 1:
                children_results = []
                with ThreadPoolExecutor(max_workers=branching_factor) as ex:
                    futures = [ex.submit(gen_one_branch, j) for j in range(branching_factor)]
                    for f in as_completed(futures):
                        children_results.append(f.result())
            else:
                children_results = [gen_one_branch(j) for j in range(branching_factor)]

            # Process results
            pruned_count = sum(1 for r in children_results if r is None)
            if pruned_count > 0:
                print(f"    [TAP] {pruned_count}/{branching_factor} branches pruned (off-topic or invalid)")
            for res in children_results:
                if res is None:
                    continue
                child, feedback = res
                node.children.append(child)
                prompt_preview = feedback.prompt[:120].replace('\n', ' ')
                response_preview = (child.response or "")[:200].replace('\n', ' ')
                print(f"    [TAP] score={child.score}, prompt=\"{prompt_preview}...\"")
                print(f"      Response: \"{response_preview}{'...' if len(child.response or '') > 200 else ''}\"")


                # Early stop only when binary judge confirms success
                if child.success:
                    print(f"  [TAP] SUCCESS at depth {iteration + 1}! score={child.score}")
                    return (feedback.prompt, True, [child])

                next_nodes.append(child)
                if (child.score or 0) > best_score:
                    best_score = child.score or 0
                    best_prompt = feedback.prompt

        # Phase 2 prune: keep top `width` by score
        next_nodes.sort(key=lambda n: (n.score or 0), reverse=True)
        if len(next_nodes) > width:
            print(f"  [TAP] width pruning: {len(next_nodes)} -> {width} nodes")
        current_nodes = next_nodes[:width]
        if not current_nodes:
            print(f"  [TAP] no surviving nodes, stopping")
            break

    return (best_prompt, found_success, current_nodes)


# ---------------------------------------------------------------------------
# TAPAttack: BaseAttack interface for PIArena registry
# ---------------------------------------------------------------------------

@register_attack
class TAPAttack(BaseAttack):
    """
    TAP attack adapted for PIArena prompt injection.

    Uses an attacker LLM to refine injection prompts in a tree structure
    with branching and pruning, evaluating each candidate through a defense.
    """

    name = "tap"
    DEFAULT_CONFIG = {
        "root_nodes": 1,
        "branching_factor": 3,
        "width": 5,
        "depth": 5,
        "temperature": 1.0,
        "max_tokens": 1024,
        "inject_position": "end",
        "pre_test": True,
        "evaluator_temperature": 0.0,
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
        evaluator_llm=None,
        inject_position=None,
        **kwargs,
    ):
        """
        Run TAP attack for one sample.

        Args:
            context: Original document context.
            injected_task: The task we want to inject.
            target_inst: User's visible question.
            llm: Target LLM (Model instance).
            attacker_llm: Attacker LLM (Model instance). Falls back to llm.
            defense: Defense instance to evaluate against.
            evaluator_llm: Evaluator LLM for on-topic/scoring. Falls back to attacker_llm.
            inject_position: Where to inject ("end", "start", "random").

        Returns:
            Injected context string (context with best attack prompt embedded).
        """
        if attacker_llm is None:
            attacker_llm = llm
        if evaluator_llm is None:
            evaluator_llm = attacker_llm
        if inject_position is None:
            inject_position = self.config.get("inject_position", "end")

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

        def evaluator_chat(messages: List[Dict[str, str]]) -> str:
            return evaluator_llm.query(
                messages,
                max_new_tokens=128,
                temperature=self.config.get("evaluator_temperature", 0.0),
            )

        def target_query(attack_prompt: str) -> str:
            injected_context = inject(context, attack_prompt, inject_position=inject_position)
            result = defense.get_response(
                target_inst=target_inst,
                context=injected_context,
                llm=llm,
            )
            return result.get("response", "")

        def judge_success_fn(response: str) -> bool:
            return refinement_judge(
                response=response,
                task_prompt=injected_task,
                evaluator_chat=evaluator_chat,
            )

        # --- Pre-test: try injected_task directly before refinement ---
        if self.config.get("pre_test", True):
            print("[TAP] Pre-test: trying direct injection before refinement...")
            pre_response = target_query(injected_task)
            if judge_success_fn(pre_response):
                print("[TAP] Pre-test SUCCESS: direct injection already works, skipping refinement")
                return inject(context, injected_task, inject_position=inject_position)
            print("[TAP] Pre-test failed, starting refinement")

        # --- Run TAP ---
        best_prompt, success, _ = run_tap_for_sample(
            goal=injected_task,
            context_tail=context_tail,
            user_task=target_inst,
            attacker_chat=attacker_chat,
            target_query=target_query,
            judge_success=judge_success_fn,
            evaluator_chat=evaluator_chat,
            root_nodes=self.config.get("root_nodes", 1),
            branching_factor=self.config.get("branching_factor", 3),
            width=self.config.get("width", 5),
            depth=self.config.get("depth", 5),
        )

        status = "SUCCESS" if success else "FAILED"
        print(f"[TAP] {status}")
        if best_prompt:
            print(f"[TAP] Final prompt:\n{best_prompt}")
            return inject(context, best_prompt, inject_position=inject_position)
        print("[TAP] No valid prompt generated, returning original context")
        return context
