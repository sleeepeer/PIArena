"""
PIArena Defense Adapter for AgentDojo Pipeline

This module provides an adapter to use PIArena defenses within AgentDojo's
agent pipeline. It filters tool outputs using PIArena's defense execute() method.

Defense Types Supported:
    - Detection (PromptGuard, DataSentinel, PIGuard, AttentionTracker):
      Empty context if injection detected
    - Sanitization (DataFilter, PISanitizer):
      Return cleaned context
    - Hybrid (PromptArmor, PromptLocate):
      Return cleaned context (with detection info)

Usage:
    Set environment variables before running:
        export PIARENA_DEFENSE=datafilter
        export PIARENA_PATH=/path/to/PIArena  # optional, auto-detected
"""

import json
import sys
import os
from collections.abc import Sequence

import torch

# Add PIArena to path
PIARENA_PATH = os.environ.get("PIARENA_PATH")
if PIARENA_PATH is None:
    # Try to auto-detect: assume agentdojo is inside PIArena/agents/
    _current = os.path.dirname(os.path.abspath(__file__))
    # Go up from agents/agentdojo/src/agentdojo/agent_pipeline to PIArena root
    _potential_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current)))))
    if os.path.exists(os.path.join(_potential_path, "piarena")):
        PIARENA_PATH = _potential_path

if PIARENA_PATH and PIARENA_PATH not in sys.path:
    sys.path.insert(0, PIARENA_PATH)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.inference_utils import parse
from agentdojo.types import ChatMessage, text_content_block_from_string
from agentdojo.functions_runtime import Env, EmptyEnv, FunctionsRuntime


def recursive_defense(obj, defense, target_inst: str):
    """
    Apply defense recursively to the input object.
    
    The input object can be a dict, list, or string.
    For strings, apply the defense's execute() method.
    For dicts/lists, recurse into nested structures.
    
    Args:
        obj: The object to apply defense to (dict, list, or string)
        defense: The PIArena defense instance with execute() method
        target_inst: The user instruction/query for context
        
    Returns:
        The defended/cleaned object with same structure as input
    """
    if isinstance(obj, dict):
        return {k: recursive_defense(v, defense, target_inst) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_defense(v, defense, target_inst) for v in obj]
    elif isinstance(obj, str):
        # Apply PIArena defense to string content
        result = defense.execute(target_inst=target_inst, context=obj)
        
        # Get cleaned context based on defense type
        if "cleaned_context" in result:
            cleaned = result["cleaned_context"]
            if isinstance(cleaned, (dict, list)):
                return json.dumps(cleaned, indent=2, ensure_ascii=False)
            return str(cleaned)
        elif result.get("detect_flag"):
            # Detection-based defense detected injection - return empty
            return ""
        else:
            # No change needed
            return obj
    else:
        return obj


class PIArenaDefenseAdapter(BasePipelineElement):
    """
    AgentDojo pipeline element that filters tool outputs using PIArena defenses.

    This adapter intercepts tool messages in the conversation and applies
    PIArena's defense mechanism to filter potentially malicious content.

    The defense to use is read from the PIARENA_DEFENSE environment variable.
    Default is 'datafilter' if not set.
    """

    def __init__(self, defense_name: str = None, defense_config: dict = None):
        # Get defense name from env or parameter
        self.defense_name = defense_name or os.environ.get("PIARENA_DEFENSE", "datafilter")
        self.defense_config = defense_config

        # Lazy load defense to avoid import issues
        self._defense = None
        print(f"[PIArenaDefenseAdapter] Will use defense: {self.defense_name}")

    def _get_defense(self):
        """Lazy load the defense."""
        if self._defense is None:
            # Initialize CUDA properly before loading defense models
            # This ensures consistent device handling when running in AgentDojo subprocess
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.set_device(0)
                print(f"[PIArenaDefenseAdapter] CUDA initialized, using device: cuda:0")
            
            from piarena.defenses import get_defense
            self._defense = get_defense(self.defense_name, self.defense_config)
            print(f"[PIArenaDefenseAdapter] Loaded defense: {self.defense_name}")
        return self._defense

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """
        Process messages and filter tool outputs.

        Only processes if the last message is a tool message.
        Applies defense recursively to all tool messages.
        """

        # Only filter if the last message is a tool message
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args

        # # Extract user instruction for context
        # user_instruction = ""
        # for m in messages:
        #     if m["role"] == "user":
        #         content = m.get("content", [])
        #         if isinstance(content, list) and len(content) > 0:
        #             first_content = content[0]
        #             if isinstance(first_content, dict):
        #                 user_instruction = first_content.get("content", "")
        #             else:
        #                 user_instruction = str(first_content)
        #         elif isinstance(content, str):
        #             user_instruction = content
        #         break

        # # Get the defense
        # defense = self._get_defense()

        # # Apply defense to all tool messages
        # for msg in messages:
        #     if msg["role"] == "tool" and "content" in msg:
        #         try:
        #             # Extract the raw tool output
        #             raw_data = msg["content"][0]["content"]

        #             # Parse the data (could be JSON or plain text)
        #             json_data = parse(raw_data)

        #             # Apply defense recursively to all string content
        #             cleaned = recursive_defense(
        #                 json_data,
        #                 defense=defense,
        #                 target_inst=user_instruction
        #             )
        #             cleaned_str = json.dumps(cleaned, indent=2, ensure_ascii=False)

        #             # Log the filtering
        #             print("\nUser instruction:", user_instruction)
        #             print("\n---------------------------------------------------------")
        #             print("Tool call result (raw):")
        #             print(raw_data)
        #             print("\n\nTool call result (cleaned by PIArena defense):")
        #             print(cleaned_str)

        #             # Update the message content with cleaned data
        #             msg["content"] = [text_content_block_from_string(cleaned_str)]

        #         except Exception as e:
        #             print(f"[PIArenaDefenseAdapter] Skipped cleaning due to error: {e}")
        #             import traceback
        #             traceback.print_exc()
        #             continue

        # Only clean the LAST tool message
        msg = messages[-1]
        defense = self._get_defense()
        if msg["role"] == "tool" and "content" in msg:
            try:
                # extract user instruction (same logic as before)
                user_instruction = ""
                for m in messages:
                    if m["role"] == "user":
                        user_instruction = m["content"][0]["content"]
                        break
    
                raw_data = msg["content"][0]["content"]
    
                json_data = parse(raw_data)
                cleaned = recursive_defense(
                    json_data,
                    defense=defense,
                    target_inst=user_instruction,
                )
                cleaned_str = json.dumps(cleaned, indent=2, ensure_ascii=False)
    
                print("\nUser instruction:", user_instruction)
                print("\n\n---------------------------------------------------------")
                print("Tool call result (raw):")
                print(raw_data)
                print("\n\n---------------------------------------------------------")
                print("Tool call result (cleaned):")
                print(cleaned_str)
    
                # Replace ONLY the last tool message's content
                msg["content"] = [text_content_block_from_string(cleaned_str)]
    
            except Exception as e:
                print(f"[PIArenaDefenseAdapter] Skipped cleaning due to error: {e}")

        return query, runtime, env, messages, extra_args
