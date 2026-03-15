import copy
import gc

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
import json

from .utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks, mellowmax, check_success
from .gcg import GCGConfig, GCGResult

class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = [] # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]
    
    def log_buffer(self, tokenizer, output, target):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}\ntarget: {target}\noutput: {output}"
        print(message)

def sample_ids_from_grad(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized 
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization
    
    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids

def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids) 
            token ids 
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer
    
    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
           filtered_ids.append(ids[i]) 
    
    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )
    
    return torch.stack(filtered_ids)

class GCG:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.prefix_cache = None
        self.attn_prefix_cache = None

        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            print(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            print("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            print("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    
    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
        injected_task: str,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
    
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
    
        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values
        
        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        success = False

        for step in tqdm(range(config.num_steps)):
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids) 
            # print(optim_ids_onehot_grad.shape)
            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences 
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                loss = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds)
                # print(loss.shape)
                # exit(0)
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            if config.early_stop and (step % 10 == 0 or float(current_loss) < 1):
                tmp_messages = copy.deepcopy(messages)
                for i in range(len(tmp_messages)):
                    if "{optim_str}" in tmp_messages[i]["content"]:
                        tmp_messages[i]["content"] = tmp_messages[i]["content"].replace("{optim_str}", optim_str)
                        break
                input = tokenizer.apply_chat_template(tmp_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                output = model.generate(input, attention_mask=torch.ones_like(input), do_sample=False, max_new_tokens=20)
                output = tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]

                buffer.log_buffer(tokenizer, output, target)                

                if config.early_stop and check_success(output, target, injected_task):
                    print("Early stopping due to finding a perfect match.") 
                    self.stop_flag = True
                    success = True

            if self.stop_flag:
                break
                
        min_loss_index = losses.index(min(losses)) if not success else -1

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            success=success,
            steps=step,
        )

        return result
    
    def run_attn(
        self,
        messages: Union[str, List[dict]],
        attn_messages: Union[str, List[dict]],
        attn_str: str,
        target: str,
        injected_task: str,
        attn_config: dict = None,
    ) -> GCGResult:
        """Run GCG optimization with attention-based loss.
        
        Args:
            messages: The conversation to use for GCG loss optimization
            attn_messages: The conversation to use for attention loss computation
            attn_str: The string in attn_messages whose attention weights should be minimized
            target: The target generation
            injected_task: The injected task description for success checking
            attn_config: Configuration for attention loss (beta weight for combining losses)
        
        Returns:
            A GCGResult object that contains losses and the optimized strings.
        """
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if "{optim_str}" in attn_str:
            attn_str = attn_str.replace("{optim_str}", "")
        
        # Set default attention config
        if attn_config is None:
            attn_config = {"beta": 1.0}  # Always averages across all layers
        
        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
    
        # Process messages for GCG loss
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
    
        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + " {optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized for GCG loss
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Process attn_messages for attention loss
        if isinstance(attn_messages, str):
            attn_template = attn_messages
        else:
            attn_messages = copy.deepcopy(attn_messages)
            attn_template = tokenizer.apply_chat_template(attn_messages, tokenize=False, add_generation_prompt=True) 

        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and attn_template.startswith(tokenizer.bos_token):
            attn_template = attn_template.replace(tokenizer.bos_token, "")
        attn_before_str, attn_after_str = attn_template.split("{optim_str}")

        # Tokenize everything for attention computation
        attn_before_ids = tokenizer([attn_before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        attn_after_ids = tokenizer([attn_after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Find attention string token indices in the attn_before_str
        attn_str_ids = tokenizer([attn_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        attn_indices = self._find_attn_str_indices(attn_before_ids, attn_str_ids[:, 1:])
        
        if attn_indices is None:
            raise ValueError(f"Attention string '{attn_str}' not found in the attn_messages")

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]
        attn_before_embeds, attn_after_embeds = [embedding_layer(ids) for ids in (attn_before_ids, attn_after_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values
                
                # Also compute attention prefix cache
                attn_output = model(inputs_embeds=attn_before_embeds, use_cache=True)
                self.attn_prefix_cache = attn_output.past_key_values
        
        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds
        self.attn_before_embeds = attn_before_embeds
        self.attn_after_embeds = attn_after_embeds
        self.attn_indices = attn_indices
        self.attn_config = attn_config

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        success = False

        for step in tqdm(range(config.num_steps)):
            # Compute the token gradient with attention loss
            optim_ids_onehot_grad = self.compute_token_gradient_attn(optim_ids) 
            
            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences with attention loss
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                loss = find_executable_batch_size(self.compute_candidates_loss_attn, batch_size)(sampled_ids)
                
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            if config.early_stop and (step % 10 == 0 or float(current_loss) < 1):
                tmp_messages = copy.deepcopy(messages)
                for i in range(len(tmp_messages)):
                    if "{optim_str}" in tmp_messages[i]["content"]:
                        tmp_messages[i]["content"] = tmp_messages[i]["content"].replace("{optim_str}", optim_str)
                        break
                input = tokenizer.apply_chat_template(tmp_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                output = model.generate(input, attention_mask=torch.ones_like(input), do_sample=False, max_new_tokens=20)
                output = tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]

                buffer.log_buffer(tokenizer, output, target)                

                if config.early_stop and check_success(output, target, injected_task):
                    print("Early stopping due to finding a perfect match.") 
                    self.stop_flag = True
                    success = True

            if self.stop_flag:
                break
                
        min_loss_index = losses.index(min(losses)) if not success else -1

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            success=success,
            steps=step,
        )

        return result
    
    def run_attn_only(
        self,
        attn_messages: Union[str, List[dict]],
        attn_str: str,
        injected_task: str,
        attn_config: dict = None,
    ) -> GCGResult:
        """Run GCG optimization with attention-only loss (no target generation loss).
        
        Args:
            attn_messages: The conversation to use for attention loss computation
            attn_str: The string in attn_messages whose attention weights should be minimized
            injected_task: The injected task description for success checking
            attn_config: Configuration for attention loss
        
        Returns:
            A GCGResult object that contains losses and the optimized strings.
        """
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if "{optim_str}" in attn_str:
            attn_str = attn_str.replace("{optim_str}", "")
        
        # Set default attention config
        if attn_config is None:
            attn_config = {"beta": 1.0}
        
        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Process attn_messages for attention loss
        if isinstance(attn_messages, str):
            attn_template = attn_messages
        else:
            attn_messages = copy.deepcopy(attn_messages)
            attn_template = tokenizer.apply_chat_template(attn_messages, tokenize=False, add_generation_prompt=True) 

        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and attn_template.startswith(tokenizer.bos_token):
            attn_template = attn_template.replace(tokenizer.bos_token, "")
        attn_before_str, attn_after_str = attn_template.split("{optim_str}")

        # Tokenize everything for attention computation
        attn_before_ids = tokenizer([attn_before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        attn_after_ids = tokenizer([attn_after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Find attention string token indices in the attn_before_str
        attn_str_ids = tokenizer([attn_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        attn_indices = self._find_attn_str_indices(attn_before_ids, attn_str_ids[:, 1:])
        
        if attn_indices is None:
            raise ValueError(f"Attention string '{attn_str}' not found in the attn_messages")

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        attn_before_embeds, attn_after_embeds = [embedding_layer(ids) for ids in (attn_before_ids, attn_after_ids)]

        # Compute the attention prefix cache
        if config.use_prefix_cache:
            with torch.no_grad():
                attn_output = model(inputs_embeds=attn_before_embeds, use_cache=True)
                self.attn_prefix_cache = attn_output.past_key_values
        
        self.attn_before_embeds = attn_before_embeds
        self.attn_after_embeds = attn_after_embeds
        self.attn_indices = attn_indices
        self.attn_config = attn_config

        # Initialize the attack buffer for attention-only optimization
        buffer = self.init_buffer_attn_only()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        success = False

        for step in tqdm(range(config.num_steps)):
            # Compute the token gradient with attention loss only
            optim_ids_onehot_grad = self.compute_token_gradient_attn_only(optim_ids) 
            
            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute attention loss only on all candidate sequences
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                loss = find_executable_batch_size(self.compute_candidates_loss_attn_only, batch_size)(sampled_ids)
                # loss = self.compute_candidates_loss_attn_only(batch_size, sampled_ids)
                
                current_loss = loss.min().item()
                print("Attention loss: ", current_loss)
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            if config.early_stop and (step % 10 == 0 or float(current_loss) < 1):
                # For attention-only, we can't easily check success without a target
                # So we'll just log the current optimization string
                print(f"Step {step}: Current optimized string: {optim_str}")

            if self.stop_flag:
                break
                
        min_loss_index = losses.index(min(losses)) if not success else -1

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            success=success,
            steps=step,
        )

        return result
    
    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        print(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
                
        else: # assume list
            if (len(config.optim_str_init) != config.buffer_size):
                print(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                print("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size) 

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer, "", "")

        print("Initialized attack buffer.")
        
        return buffer
    
    def init_buffer_attn_only(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        print(f"Initializing attention-only attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
                
        else: # assume list
            if (len(config.optim_str_init) != config.buffer_size):
                print(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                print("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size) 

        # Compute the attention loss on the initial buffer entries
        init_buffer_losses = find_executable_batch_size(self.compute_candidates_loss_attn_only, true_buffer_size)(init_buffer_ids)
        # init_buffer_losses = self.compute_candidates_loss_attn_only(1, init_buffer_ids)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer, "", "")

        print("Initialized attention-only attack buffer.")
        
        return buffer
    
    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized 
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, use_cache=True)
        else:
            input_embeds = torch.cat([self.before_embeds, optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds)

        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift-1:-1, :].contiguous() # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad
    
    def compute_candidates_loss(
        self,
        search_batch_size: int, 
        input_embeds: Tensor, 
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch, use_cache=True)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)
                
                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                # if self.config.early_stop:
                #     if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                #         self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def _find_attn_str_indices(self, before_ids: Tensor, attn_str_ids: Tensor) -> Optional[Tensor]:
        """Find the token indices of the attention string in the before_ids sequence.
        
        Args:
            before_ids: Token IDs of the text before the optimization string
            attn_str_ids: Token IDs of the attention string to find
            
        Returns:
            Tensor of indices where the attention string appears, or None if not found
        """
        before_seq = before_ids.squeeze(0)
        attn_seq = attn_str_ids.squeeze(0)
        
        if len(attn_seq) > len(before_seq):
            return None
            
        # Find all possible starting positions
        for i in range(len(before_seq) - len(attn_seq) + 1):
            if torch.equal(before_seq[i:i+len(attn_seq)], attn_seq):
                return torch.arange(i, i + len(attn_seq), device=before_ids.device)
        
        return None

    def compute_token_gradient_attn(self, optim_ids: Tensor) -> Tensor:
        """Computes the gradient of the combined GCG + attention loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized 
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        # Compute GCG loss using messages
        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, use_cache=True)
        else:
            input_embeds = torch.cat([self.before_embeds, optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds)

        logits = output.logits

        # Compute original GCG loss
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift-1:-1, :].contiguous()
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            gcg_loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            gcg_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Compute attention loss using attn_messages
        # Create input for attention computation
        attn_input_embeds = torch.cat([self.attn_before_embeds, optim_embeds, self.attn_after_embeds], dim=1)
        
        # Get hidden states from the attention input (similar to GCG loss computation)
        if self.attn_prefix_cache:
            attn_output = model(inputs_embeds=torch.cat([optim_embeds, self.attn_after_embeds], dim=1), 
                               past_key_values=self.attn_prefix_cache, use_cache=True, output_hidden_states=True)
        else:
            attn_output = model(inputs_embeds=attn_input_embeds, output_hidden_states=True)
        
        attn_hidden_states = attn_output.hidden_states
        attn_loss = self._compute_attention_loss(attn_hidden_states, attn_input_embeds.shape[1], int(self.attn_indices[-1]))
        
        # Combine losses
        beta = self.attn_config.get("beta", 1.0)
        total_loss = gcg_loss + beta * attn_loss

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[total_loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def compute_token_gradient_attn_only(self, optim_ids: Tensor) -> Tensor:
        """Computes the gradient of the attention loss only w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized 
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        # Compute attention loss using attn_messages
        # Create input for attention computation
        attn_input_embeds = torch.cat([self.attn_before_embeds, optim_embeds, self.attn_after_embeds], dim=1)
        
        # Get hidden states from the attention input (similar to GCG loss computation)
        if self.attn_prefix_cache:
            attn_output = model(inputs_embeds=torch.cat([optim_embeds, self.attn_after_embeds], dim=1), 
                               past_key_values=self.attn_prefix_cache, use_cache=True, output_hidden_states=True)
        else:
            attn_output = model(inputs_embeds=attn_input_embeds, output_hidden_states=True)
        
        attn_hidden_states = attn_output.hidden_states
        attn_loss = self._compute_attention_loss(attn_hidden_states, attn_input_embeds.shape[1], int(self.attn_indices[-1]))
        
        # Debug: Check if loss is NaN or doesn't require grad
        # print(f"Debug: attn_loss = {attn_loss}, requires_grad = {attn_loss.requires_grad}, is_nan = {torch.isnan(attn_loss)}")
        
        if torch.isnan(attn_loss):
            print("Warning: Attention loss is NaN, using zero gradient")
            optim_ids_onehot_grad = torch.zeros_like(optim_ids_onehot)
        elif not attn_loss.requires_grad:
            print("Warning: Attention loss doesn't require grad, using zero gradient")
            optim_ids_onehot_grad = torch.zeros_like(optim_ids_onehot)
        else:
            optim_ids_onehot_grad = torch.autograd.grad(outputs=[attn_loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def _compute_attention_loss(self, hidden_states: tuple, input_seq_len: int, last_attn_index: int) -> Tensor:
        """Compute attention loss for the last input token attending to the attention string.
        
        Args:
            hidden_states: Hidden states from all layers
            input_seq_len: Length of the input sequence
            last_attn_index: Index of the last attention token
            
        Returns:
            Attention loss scalar (averaged across all layers)
        """
        from .attention_utils import get_layer_attention_weights, get_position_ids_and_attention_mask, infer_model_type
        
        model = self.model
        model_type = infer_model_type(model)
        
        # Get number of layers
        try:
            num_layers = len(model.model.layers)
        except:
            num_layers = len(model.model.language_model.layers)
        
        layer_attention_losses = []
        
        # Get position IDs and attention mask (without no_grad)
        position_ids, attention_mask = get_position_ids_and_attention_mask(model, hidden_states)
        
        # Compute attention loss for each layer
        for layer_idx in range(num_layers):
            # Get attention weights for this layer (without no_grad)
            # We want attention from the last input token to the attention string
            attn_weights = get_layer_attention_weights(
                model,
                hidden_states,
                layer_idx,
                position_ids,
                attention_mask,
                attribution_start=input_seq_len,  # Last input token
                attribution_end=input_seq_len + 1,  # Only last input token
                model_type=model_type,
            )
            
            # print(f"Debug layer {layer_idx}: attn_weights shape = {attn_weights.shape}, attn_indices = {self.attn_indices}")
            # print(f"Debug layer {layer_idx}: attn_weights requires_grad = {attn_weights.requires_grad}")
            # print(f"Debug layer {layer_idx}: attn_weights min/max = {attn_weights.min()}/{attn_weights.max()}")
            
            # attn_weights shape: [num_heads, 1, input_seq_len] (for last input token attending to all input tokens)
            # We want to minimize attention to the attention string tokens
            if len(self.attn_indices) > 0 and self.attn_indices.max() < attn_weights.shape[-1]:
                attn_to_str = attn_weights[:, :, :, self.attn_indices].mean()  # Average over heads and attention string tokens
                # print(f"Debug layer {layer_idx}: attn_to_str = {attn_to_str}, requires_grad = {attn_to_str.requires_grad}")
                layer_attention_losses.append(attn_to_str)
            else:
                print(f"Warning: attn_indices {self.attn_indices} out of bounds for attn_weights shape {attn_weights.shape}")
                # Use a small positive value to avoid NaN
                layer_attention_losses.append(torch.tensor(0.01, device=model.device, requires_grad=True))
        
        # Average across all layers
        if layer_attention_losses:
            avg_attention_loss = torch.stack(layer_attention_losses).mean()
        else:
            avg_attention_loss = torch.tensor(0.01, device=model.device, requires_grad=True)
        
        return avg_attention_loss

    def compute_candidates_loss_attn(self, batch_size: int, sampled_ids: Tensor) -> Tensor:
        """Computes the combined GCG + attention loss on all candidate token id sequences.

        Args:
            batch_size : int
                the number of candidate sequences to evaluate in a given batch
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                the candidate sequences to evaluate
        """
        all_loss = []
        embedding_layer = self.embedding_layer

        for i in range(0, sampled_ids.shape[0], batch_size):
            with torch.no_grad():
                sampled_ids_batch = sampled_ids[i:i+batch_size]
                current_batch_size = sampled_ids_batch.shape[0]

                # Compute GCG loss using messages
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids_batch),
                        self.after_embeds.repeat(current_batch_size, 1, 1),
                        self.target_embeds.repeat(current_batch_size, 1, 1),
                    ], dim=1)
                    
                    prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[j]] for j in range(len(self.prefix_cache))]
                    outputs = self.model(inputs_embeds=input_embeds, past_key_values=prefix_cache_batch, use_cache=True)
                else:
                    input_embeds = torch.cat([
                        self.before_embeds.repeat(current_batch_size, 1, 1),
                        embedding_layer(sampled_ids_batch),
                        self.after_embeds.repeat(current_batch_size, 1, 1),
                        self.target_embeds.repeat(current_batch_size, 1, 1),
                    ], dim=1)
                    outputs = self.model(inputs_embeds=input_embeds)

                logits = outputs.logits

                # Compute original GCG loss
                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)
                
                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    gcg_loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                else:
                    gcg_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

                gcg_loss = gcg_loss.view(current_batch_size, -1).mean(dim=-1)

                # Compute attention loss for each sample in the batch using attn_messages
                attn_losses = []
                for b in range(current_batch_size):
                    # Create attention input for this sample
                    if self.attn_prefix_cache:
                        attn_input_embeds = torch.cat([
                            embedding_layer(sampled_ids_batch[b:b+1]),
                            self.attn_after_embeds
                        ], dim=1)
                        attn_prefix_cache_single = [[x[0:1] for x in self.attn_prefix_cache[j]] for j in range(len(self.attn_prefix_cache))]
                        attn_output = self.model(inputs_embeds=attn_input_embeds, 
                                               past_key_values=attn_prefix_cache_single, use_cache=True, output_hidden_states=True)
                    else:
                        attn_input_embeds = torch.cat([
                            self.attn_before_embeds,
                            embedding_layer(sampled_ids_batch[b:b+1]),
                            self.attn_after_embeds
                        ], dim=1)
                        attn_output = self.model(inputs_embeds=attn_input_embeds, output_hidden_states=True)
                    
                    # Get hidden states and compute attention loss
                    attn_hidden_states = attn_output.hidden_states
                    attn_loss = self._compute_attention_loss(attn_hidden_states, attn_input_embeds.shape[1], int(self.attn_indices[-1]))
                    attn_losses.append(attn_loss)
                
                attn_losses = torch.stack(attn_losses)
                
                # Combine losses
                beta = self.attn_config.get("beta", 1.0)
                total_loss = gcg_loss + beta * attn_losses

                print(f"GCG loss: {gcg_loss.mean().item()}, Attention loss: {attn_losses.mean().item()}, Total loss: {total_loss.mean().item()}")
                
                all_loss.append(total_loss)

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def compute_candidates_loss_attn_only(self, batch_size: int, sampled_ids: Tensor) -> Tensor:
        """Computes the attention loss only on all candidate token id sequences.

        Args:
            batch_size : int
                the number of candidate sequences to evaluate in a given batch
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                the candidate sequences to evaluate
        """
        all_loss = []
        embedding_layer = self.embedding_layer

        for i in range(0, sampled_ids.shape[0], batch_size):
            with torch.no_grad():
                sampled_ids_batch = sampled_ids[i:i+batch_size]
                current_batch_size = sampled_ids_batch.shape[0]

                # Compute attention loss for each sample in the batch using attn_messages
                attn_losses = []
                for b in range(current_batch_size):
                    # Create attention input for this sample
                    if self.attn_prefix_cache:
                        attn_input_embeds = torch.cat([
                            embedding_layer(sampled_ids_batch[b:b+1]),
                            self.attn_after_embeds
                        ], dim=1)
                        attn_prefix_cache_single = [[x[0:1] for x in self.attn_prefix_cache[j]] for j in range(len(self.attn_prefix_cache))]
                        attn_output = self.model(inputs_embeds=attn_input_embeds, 
                                               past_key_values=attn_prefix_cache_single, use_cache=True, output_hidden_states=True)
                    else:
                        attn_input_embeds = torch.cat([
                            self.attn_before_embeds,
                            embedding_layer(sampled_ids_batch[b:b+1]),
                            self.attn_after_embeds
                        ], dim=1)
                        attn_output = self.model(inputs_embeds=attn_input_embeds, output_hidden_states=True)
                    
                    # Get hidden states and compute attention loss
                    attn_hidden_states = attn_output.hidden_states
                    attn_loss = self._compute_attention_loss(attn_hidden_states, attn_input_embeds.shape[1], int(self.attn_indices[-1]))
                    attn_losses.append(attn_loss)
                
                attn_losses = torch.stack(attn_losses)
                
                # print(f"Attention loss: {attn_losses.mean().item()}")
                
                all_loss.append(attn_losses)

                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    injected_task: str,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG. 

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    # logger.setLevel(getattr(logging, config.verbosity))
    
    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages, target, injected_task)
    # gc.collect()
    # torch.cuda.empty_cache()
    return result

# A wrapper around the GCG `run_multi` method that provides a simple API
def run_multi(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    multi_messages: List[Union[str, List[dict]]],
    multi_targets: List[str],
    injected_task: str = 'substring',
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG that works across multiple prompts. 

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        multi_messages: List of conversations to use for optimization.
        multi_targets: List of target generations corresponding to each conversation.
        injected_task: The injected task description for success checking.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    # logger.setLevel(getattr(logging, config.verbosity))
    
    gcg = GCG(model, tokenizer, config)
    result = gcg.run_multi(multi_messages, multi_targets, injected_task)
    # gc.collect()
    # torch.cuda.empty_cache()
    return result

# A wrapper around the GCG `run_attn` method that provides a simple API
def run_attn(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    attn_messages: Union[str, List[dict]],
    attn_str: str,
    target: str,
    injected_task: str,
    attn_config: Optional[dict] = None,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG with attention-based loss.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for GCG loss optimization.
        attn_messages: The conversation to use for attention loss computation.
        attn_str: The string in attn_messages whose attention weights should be minimized.
        target: The target generation.
        injected_task: The injected task description for success checking.
        attn_config: Configuration for attention loss (beta weight for combining losses).
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    if attn_config is None:
        attn_config = {"beta": 1.0}
    
    gcg = GCG(model, tokenizer, config)
    result = gcg.run_attn(messages, attn_messages, attn_str, target, injected_task, attn_config)
    return result

# A wrapper around the GCG `run_attn_only` method that provides a simple API
def run_attn_only(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    attn_messages: Union[str, List[dict]],
    attn_str: str,
    injected_task: str,
    attn_config: Optional[dict] = None,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using attention loss only.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        attn_messages: The conversation to use for attention loss computation.
        attn_str: The string in attn_messages whose attention weights should be minimized.
        injected_task: The injected task description for success checking.
        attn_config: Configuration for attention loss.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    if attn_config is None:
        attn_config = {"beta": 1.0}
    
    gcg = GCG(model, tokenizer, config)
    result = gcg.run_attn_only(attn_messages, attn_str, injected_task, attn_config)
    return result
    