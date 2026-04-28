from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Union

import openai
import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer

from google import genai
from google.genai import types
import anthropic


# --------------------------------------------------------------------------- #
# Provider helpers (Azure / OpenAI / Google / Anthropic)
# --------------------------------------------------------------------------- #

def load_gpt_model(openai_config_path, model_name, api_key_index=0):
    with open(openai_config_path, "r") as file:
        config = yaml.safe_load(file)["default"]
    usable_keys = []
    for item in config:
        if item.get("azure_deployment", model_name) == model_name:
            if "azure_deployment" in item:
                del item["azure_deployment"]
            usable_keys.append(item)
    client_class = usable_keys[api_key_index]["client_class"]
    del usable_keys[api_key_index]["client_class"]
    return eval(client_class)(**usable_keys[api_key_index])


def get_openai_completion_with_retry(client, sleepsec=10, **kwargs) -> str:
    while 1:
        try:
            return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            if "400" in str(e):
                return "OpenAI Rejected"
            print("OpenAI API error:", e, "sleeping for", sleepsec, "seconds", flush=True)
            time.sleep(sleepsec)


class OpenAIModel:
    def __init__(self, model_name_or_path):
        model_name = model_name_or_path.split("/")[-1]
        self.model_name = model_name
        openai_config_path = f"configs/openai_configs/{model_name}.yaml"
        self.config = yaml.safe_load(open(openai_config_path, "r"))
        self.client = openai.OpenAI(api_key=self.config["api_key"])

    def query(self, messages: Union[str, List[Dict[str, str]]], **kwargs):
        return get_openai_completion_with_retry(
            self.client,
            messages=messages,
            model=self.config["model"],
        )


class GoogleModel:
    def __init__(self, model_name_or_path):
        model_name = model_name_or_path.split("/")[-1]
        self.model_name = model_name
        google_config_path = f"configs/google_configs/{model_name}.yaml"
        self.config = yaml.safe_load(open(google_config_path, "r"))
        self.client = genai.Client(api_key=self.config["api_key"])

    def query(self, messages: Union[str, List[Dict[str, str]]], **kwargs):
        input_contents = " ".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        return self.client.models.generate_content(
            model=self.config["model"],
            contents=input_contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=self.config["thinking_level"])
            ),
        ).text


class AnthropicModel:
    def __init__(self, model_name_or_path):
        model_name = model_name_or_path.split("/")[-1]
        self.model_name = model_name
        anthropic_config_path = f"configs/anthropic_configs/{model_name}.yaml"
        self.config = yaml.safe_load(open(anthropic_config_path, "r"))
        self.client = anthropic.Anthropic(api_key=self.config["api_key"])

    def query(self, messages: Union[str, List[Dict[str, str]]], **kwargs):
        anthropic_messages = []
        system_prompt = ""
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            else:
                anthropic_messages.append(message)
        return self.client.messages.create(
            model=self.config["model"],
            max_tokens=1000,
            messages=anthropic_messages,
            system=system_prompt,
        ).content[0].text


# --------------------------------------------------------------------------- #
# vLLM backend
# --------------------------------------------------------------------------- #

class _VLLMBackend:
    """Thin wrapper around `vllm.LLM` shared by `Model.query` / `Model.batch_query`."""

    def __init__(
        self,
        model_name_or_path: str,
        tp_size: int = 1,
        gpu_mem: float = 0.85,
        max_model_len: int = 8192,
        max_num_seqs: int = 256,
        enforce_eager: bool = False,
    ):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_mem,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=enforce_eager,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        sampling = self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        outputs = self.llm.generate(prompts, sampling, use_tqdm=False)
        return [out.outputs[0].text for out in outputs]


# --------------------------------------------------------------------------- #
# Reasoning-token helpers
# --------------------------------------------------------------------------- #

THINK_CLOSE = "</think>"


def _split_reasoning(text: str) -> tuple[str, Optional[str]]:
    """Return (final_answer, reasoning_or_None) by splitting on the last `</think>`."""
    if THINK_CLOSE not in text:
        return text, None
    thinking, _, final = text.rpartition(THINK_CLOSE)
    return final.lstrip(), thinking.strip()


# --------------------------------------------------------------------------- #
# The Model facade
# --------------------------------------------------------------------------- #

_PROVIDER_KEYWORDS = ("azure", "openai", "google", "anthropic")


def _resolve_backend(model_name_or_path: str, requested: str) -> str:
    """Pick a concrete backend label.

    Provider strings (azure / openai / google / anthropic) are detected from the
    model id. Otherwise the caller's `requested` value wins, with `hf` as the
    default for HF-style identifiers.
    """
    lower = model_name_or_path.lower()
    for kw in _PROVIDER_KEYWORDS:
        if kw in lower:
            return kw
    if requested == "vllm":
        return "vllm"
    return "hf"


class Model:
    """Unified interface over Azure/OpenAI/Google/Anthropic, vLLM, and HF backends.

        m = Model("Qwen/Qwen3.5-9B")                       # HF (default)
        m = Model("Qwen/Qwen3.5-9B", backend="vllm",        # vLLM with tunables
                  tp_size=1, gpu_mem=0.85,
                  max_model_len=8192, max_num_seqs=256,
                  enforce_eager=False)
        m.query(messages, max_new_tokens=...) -> str
        m.batch_query([messages, ...], ...) -> List[str]

    After a query, `m.last_reasoning` holds the stripped `<think>...</think>`
    block (if any) or None.
    """

    def __init__(
        self,
        model_name_or_path: str,
        backend: str = "hf",
        tp_size: int = 1,
        gpu_mem: float = 0.85,
        max_model_len: int = 8192,
        max_num_seqs: int = 256,
        enforce_eager: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.backend = _resolve_backend(model_name_or_path, backend)
        self.last_reasoning: Optional[str] = None

        if self.backend == "azure":
            model_name = model_name_or_path.split("/")[-1]
            self.model_name = model_name
            openai_config_path = f"configs/azure_configs/{model_name}.yaml"
            self.model = load_gpt_model(openai_config_path, model_name, 0)
            self.tokenizer = None
        elif self.backend == "openai":
            self.model = OpenAIModel(model_name_or_path)
            self.tokenizer = None
        elif self.backend == "google":
            self.model = GoogleModel(model_name_or_path)
            self.tokenizer = None
        elif self.backend == "anthropic":
            self.model = AnthropicModel(model_name_or_path)
            self.tokenizer = None
        elif self.backend == "vllm":
            self._vllm = _VLLMBackend(
                model_name_or_path,
                tp_size=tp_size,
                gpu_mem=gpu_mem,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                enforce_eager=enforce_eager,
            )
            self.model = self._vllm.llm
            self.tokenizer = self._vllm.tokenizer
        elif self.backend == "hf":
            self._load_hf()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_hf(self):
        while True:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    use_fast=True,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN"),
                    cache_dir=os.getenv("HF_HOME"),
                )
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        device_map="auto",
                        dtype="auto",
                        token=os.getenv("HF_TOKEN"),
                        cache_dir=os.getenv("HF_HOME"),
                    )
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path,
                        device_map="auto",
                        torch_dtype="auto",
                        token=os.getenv("HF_TOKEN"),
                        cache_dir=os.getenv("HF_HOME"),
                    )
                self.model.eval()
                break
            except Exception as e:
                if "429" in str(e):
                    print("Hit Hugging Face rate limit when loading model. Waiting 5 minutes...")
                    time.sleep(300)
                else:
                    raise
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not getattr(self.model.config, "is_encoder_decoder", False):
            self.tokenizer.padding_side = "left"

    @staticmethod
    def _normalize_messages(messages: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if isinstance(messages, str):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages},
            ]
        return messages

    def _format_messages_for_generation(
        self, messages: Union[str, List[Dict[str, str]]]
    ) -> str:
        normalized = self._normalize_messages(messages)
        # Pass `enable_thinking=False` when the tokenizer supports it (Qwen3+).
        # Older templates raise TypeError on the kwarg; fall back gracefully.
        try:
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                return self.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = ""
                for m in normalized:
                    role = m.get("role", "user").capitalize()
                    content = m.get("content", "")
                    prompt += f"{role}: {content}\n\n"
                prompt += "Assistant: "
                return prompt

    def query(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
    ) -> str:
        if self.backend == "azure":
            text = get_openai_completion_with_retry(
                self.model, messages=messages, model=self.model_name
            )
        elif self.backend in ("openai", "google", "anthropic"):
            text = self.model.query(messages)
        elif self.backend == "vllm":
            prompt = self._format_messages_for_generation(messages)
            effective_temp = temperature if do_sample else 0.0
            effective_top_p = top_p if do_sample else 1.0
            text = self._vllm.generate(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=effective_temp,
                top_p=effective_top_p,
            )[0]
        else:  # hf
            text = self._hf_generate_one(
                messages, max_new_tokens, temperature, do_sample, top_p
            )

        final, reasoning = _split_reasoning(text)
        self.last_reasoning = reasoning
        if reasoning is not None:
            sys.stderr.write(
                f"[reasoning-strip] think_chars={len(reasoning)} final_preview={final[:160]!r}\n"
            )
        return final

    def batch_query(
        self,
        messages_list: List[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
    ) -> List[str]:
        if not messages_list:
            return []

        if self.backend in ("azure", "openai", "google", "anthropic"):
            return [
                self.query(
                    messages=m,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                )
                for m in messages_list
            ]

        if self.backend == "vllm":
            prompts = [self._format_messages_for_generation(m) for m in messages_list]
            effective_temp = temperature if do_sample else 0.0
            effective_top_p = top_p if do_sample else 1.0
            raw = self._vllm.generate(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=effective_temp,
                top_p=effective_top_p,
            )
            finals: List[str] = []
            last: Optional[str] = None
            for text in raw:
                final, reasoning = _split_reasoning(text)
                if reasoning is not None:
                    last = reasoning
                finals.append(final)
            self.last_reasoning = last
            return finals

        return self._hf_batch_generate(
            messages_list, max_new_tokens, temperature, do_sample, top_p
        )

    def _hf_generate_one(
        self, messages, max_new_tokens, temperature, do_sample, top_p
    ) -> str:
        normalized = self._normalize_messages(messages)
        tokenized = self.tokenizer.apply_chat_template(
            normalized, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers >=5 returns BatchEncoding; older returns LongTensor.
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized["input_ids"].to(self.model.device)
            attention_mask = tokenized.get(
                "attention_mask", torch.ones_like(tokenized["input_ids"])
            ).to(self.model.device)
        else:
            input_ids = tokenized.to(self.model.device)
            attention_mask = torch.ones_like(input_ids).to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            outputs[0][len(input_ids[0]):], skip_special_tokens=True
        )

    def _hf_batch_generate(
        self, messages_list, max_new_tokens, temperature, do_sample, top_p
    ) -> List[str]:
        prompts = [self._format_messages_for_generation(m) for m in messages_list]
        tokenized = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Tokenizer is left-padded (`padding_side = "left"` in `_load_hf`), so the
        # real prompt sits at positions `[max_len - N, max_len)` while pad tokens
        # occupy `[0, max_len - N)`. Slicing by `attention_mask.sum() == N` keeps
        # the real prompt in the decoded text; we have to slice from the padded
        # input length to drop the prompt cleanly.
        prompt_len = tokenized["input_ids"].shape[1]
        finals: List[str] = []
        last: Optional[str] = None
        for output in outputs:
            text = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            final, reasoning = _split_reasoning(text)
            if reasoning is not None:
                last = reasoning
            finals.append(final)
        self.last_reasoning = last
        return finals
