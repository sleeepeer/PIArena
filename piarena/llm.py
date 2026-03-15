from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import openai
import yaml
import time

from google import genai
from google.genai import types

from typing import Union, List, Dict
import anthropic


def load_gpt_model(openai_config_path, model_name, api_key_index=0):
    with open(openai_config_path, 'r') as file: config = yaml.safe_load(file)['default']
    usable_keys = []
    for item in config:
        if item.get('azure_deployment', model_name) == model_name:
            if 'azure_deployment' in item: del item['azure_deployment']
            usable_keys.append(item)
    client_class = usable_keys[api_key_index]['client_class']
    del usable_keys[api_key_index]['client_class']
    return eval(client_class)(**usable_keys[api_key_index])

def get_openai_completion_with_retry(client, sleepsec=10, **kwargs) -> str:
    while 1:
        try: return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            if "400" in str(e):
                return "OpenAI Rejected"
            print('OpenAI API error:', e, 'sleeping for', sleepsec, 'seconds', flush=True) 
            time.sleep(sleepsec)
            
            

class GoogleModel():
    def __init__(self, model_name_or_path):
        model_name = model_name_or_path.split("/")[-1]
        self.model_name = model_name
        google_config_path = f"configs/google_configs/{model_name}.yaml"
        self.config = yaml.safe_load(open(google_config_path, 'r'))
        self.client = genai.Client(api_key=self.config['api_key'])

    def query(self, messages: Union[str, List[Dict[str, str]]], **kwargs):
        input_contents = " ".join([f"{message['role']}: {message['content']}" for message in messages])
        return self.client.models.generate_content(
            model=self.config['model'],
            contents=input_contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=self.config['thinking_level'])
            ),
        ).text
    
class AnthropicModel():
    def __init__(self, model_name_or_path):
        model_name = model_name_or_path.split("/")[-1]
        self.model_name = model_name
        anthropic_config_path = f"configs/anthropic_configs/{model_name}.yaml"
        self.config = yaml.safe_load(open(anthropic_config_path, 'r'))
        self.client = anthropic.Anthropic(api_key=self.config['api_key'])

    def query(self, messages: Union[str, List[Dict[str, str]]], **kwargs):
        anthropic_messages = []
        system_prompt = ""
        for message in messages:
            if message['role'] == 'system':
                system_prompt = message['content']
            else:
                anthropic_messages.append(message)
        return self.client.messages.create(
            model=self.config['model'],
            max_tokens=1000,
            messages=anthropic_messages,
            system=system_prompt,
        ).content[0].text
            
class Model():
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        if "azure" in model_name_or_path.lower():
            model_name = model_name_or_path.split("/")[-1]
            self.model_name = model_name
            openai_config_path = f"configs/azure_configs/{model_name}.yaml"
            api_key_index = 0
            self.model = load_gpt_model(openai_config_path, model_name, api_key_index)
            self.tokenizer = None
        elif "google" in model_name_or_path.lower():
            self.model = GoogleModel(model_name_or_path)
            self.tokenizer = None
        elif "anthropic" in model_name_or_path.lower():
            self.model = AnthropicModel(model_name_or_path)
            self.tokenizer = None
        else:
            while True:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name_or_path, 
                        use_fast=True, 
                        trust_remote_code=True, 
                        token=os.getenv("HF_TOKEN"), 
                        cache_dir=os.getenv("HF_HOME")
                    )
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name_or_path, 
                            device_map="auto", 
                            # output_attentions=output_attentions,
                            # attn_implementation="flash_attention_2",
                            # attn_implementation="sdpa",
                            dtype="auto", 
                            token=os.getenv("HF_TOKEN"), 
                            cache_dir=os.getenv("HF_HOME")
                        )
                    except Exception as e:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name_or_path, 
                            device_map="auto", 
                            torch_dtype="auto", 
                            token=os.getenv("HF_TOKEN"), 
                            cache_dir=os.getenv("HF_HOME")
                        )
                    self.model.eval()
                    break
                except Exception as e:
                    if "429" in str(e):
                        print("Hit Hugging Face rate limit when loading model. Waiting 5 minutes...")
                        time.sleep(300)
                    else:
                        raise e
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token     


    def query(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
    ):
        if "azure" in self.model_name_or_path.lower():
            generated_text = get_openai_completion_with_retry(self.model, 
                messages=messages,
                model=self.model_name, 
            )       
        elif "google" in self.model_name_or_path.lower():
            generated_text = self.model.query(messages)
        elif "anthropic" in self.model_name_or_path.lower():
            generated_text = self.model.query(messages)
        else: # HF models
            if isinstance(messages, str):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": messages}
                ]
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            attention_mask = torch.ones_like(input_ids).to(self.model.device)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
                
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                repetition_penalty=1.2
            )
            generated_text = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return generated_text