import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from ..base import BaseDefense, register_defense


@register_defense
class PIGuardDefense(BaseDefense):
    """Detection defense using the PIGuard fine-tuned classifier."""

    name = "piguard"
    DEFAULT_CONFIG = {}

    def __init__(self, config=None):
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(
            "leolee99/PIGuard", cache_dir=os.getenv("HF_HOME")
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "leolee99/PIGuard",
            cache_dir=os.getenv("HF_HOME"),
            trust_remote_code=True,
            device_map="auto",
        )
        self._detector = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
        )

    def execute(self, target_inst, context):
        result = self._detector(context)[0]
        prediction = result["label"]
        score = result["score"]

        if prediction == "benign":
            detect_flag = False
        elif prediction == "injection":
            detect_flag = True
        else:
            raise ValueError(f"Unknown prediction: {prediction}")

        return {"detect_flag": detect_flag, "score": score}

    def get_response(self, target_inst, context, llm,
                     system_prompt="You are a helpful assistant."):
        result = self.execute(target_inst, context)
        if llm is not None:
            if result["detect_flag"]:
                result["response"] = "[Warning] PIGuard detected injected prompt in the context."
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_inst}\n\n{context}"},
                ]
                result["response"] = llm.query(messages)
        else:
            result["response"] = "[PIArena] No LLM provided, response is not available."
        return result
