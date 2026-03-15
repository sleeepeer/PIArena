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

def nice_print(text):
    print_text = copy.deepcopy(text)
    print_text = print_text.replace("\\", "\\\\").replace("\n", "\\n")
    print(print_text)

@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = 128
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    saved_path: str = None

@dataclass
class GCGResult:
    best_loss: float = None
    best_string: str = None
    losses: List[float] = None
    strings: List[str] = None
    success: bool = False
    steps: int = 0
    current_output: str = None

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
            # message += f"\nloss: {loss}" + f" | string: {optim_str}\ntarget: {target}\noutput: {output}"
            message += f"\nloss: {loss}" + f"\ntarget: {target}\noutput: {output}"

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
        early_stop_condition: str,
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
        current_output = ""

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

            if config.early_stop and (step % 10 == 0 or float(current_loss) < 0.1):
                tmp_messages = copy.deepcopy(messages)
                for i in range(len(tmp_messages)):
                    if "{optim_str}" in tmp_messages[i]["content"]:
                        tmp_messages[i]["content"] = tmp_messages[i]["content"].replace("{optim_str}", optim_str)
                        break
                input = tokenizer.apply_chat_template(tmp_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                output = model.generate(input, attention_mask=torch.ones_like(input), do_sample=False, max_new_tokens=1024)
                output = tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]
                current_output = output
                buffer.log_buffer(tokenizer, output, target)                

                if config.early_stop and check_success(output, target, early_stop_condition):
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
            current_output=current_output,
        )

        return result

    def run_multi(
        self,
        multi_messages: List[Union[str, List[dict]]],
        multi_targets: List[str],
        early_stop_condition: str = 'start_with',
    ):
        """
        A multi prompt attack that runs GCG on multiple prompts in parallel and get only one result for all prompts.

        i.e., we optimize a single best string that could be used for all prompts.

        So we need to average the loss of all prompts to get the best string.
        """
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        # if config.seed is not None:
        #     set_seed(config.seed)
        #     torch.use_deterministic_algorithms(True, warn_only=True)
    
        # Process all message/target pairs
        num_prompts = len(multi_messages)
        if len(multi_targets) != num_prompts:
            raise ValueError(f"Number of messages ({num_prompts}) must match number of targets ({len(multi_targets)})")
        
        # Store processed data for each prompt
        all_before_embeds = []
        all_after_embeds = []
        all_target_embeds = []
        all_target_ids = []
        all_prefix_caches = []
        
        for i in range(num_prompts):
            messages = multi_messages[i]
            target = multi_targets[i]
            
            
            if isinstance(messages, str):
                assert "{optim_str}" in messages
                # messages = [{"role": "user", "content": messages}]
                # if "{optim_str}" not in messages:
                #     messages = messages + "{optim_str}"
                template = copy.deepcopy(messages)
            else:
                messages = copy.deepcopy(messages)
                # Append the GCG string at the end of the prompt if location not specified
                if not any(["{optim_str}" in d["content"] for d in messages]):
                    messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

                template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

            print_template = template.replace('\n', '\\n').replace('\\', '\\\\')
            print(f"Prompt {i}: {print_template}, Target: {target}")
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
            
            all_before_embeds.append(before_embeds)
            all_after_embeds.append(after_embeds)
            all_target_embeds.append(target_embeds)
            all_target_ids.append(target_ids)

            # Compute the KV Cache for tokens that appear before the optimized tokens
            if config.use_prefix_cache:
                with torch.no_grad():
                    output = model(inputs_embeds=before_embeds, use_cache=True)
                    all_prefix_caches.append(output.past_key_values)
            else:
                all_prefix_caches.append(None)
        
        # Store data for gradient computation
        self.all_target_ids = all_target_ids
        self.all_before_embeds = all_before_embeds
        self.all_after_embeds = all_after_embeds
        self.all_target_embeds = all_target_embeds
        self.all_prefix_caches = all_prefix_caches

        # Initialize the attack buffer using the first prompt's data for initialization
        # but we'll compute the actual loss across all prompts
        self.target_ids = all_target_ids[0]
        self.before_embeds = all_before_embeds[0]
        self.after_embeds = all_after_embeds[0]
        self.target_embeds = all_target_embeds[0]
        self.prefix_cache = all_prefix_caches[0]
        
        buffer = self.init_buffer_multi()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        success = False
        
        # with open(config.saved_path, mode="w", encoding="utf-8") as f:
        for step in tqdm(range(config.num_steps)):
            # print(f"Step {step}")
            # Compute the token gradient averaged across all prompts
            optim_ids_onehot_grad = self.compute_token_gradient_multi(optim_ids)
            
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

                # Compute loss on all candidate sequences across all prompts
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                
                # Create a wrapper function that matches find_executable_batch_size expectations
                def compute_loss_wrapper(batch_size_arg):
                    return self.compute_candidates_loss_multi(sampled_ids, batch_size_arg)
                
                loss = find_executable_batch_size(compute_loss_wrapper, batch_size)()
                
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            print(f" Loss: {current_loss}", end="")
            print(f"\nOptim Str: {optim_str}")
            

            # Check success on all prompts and generate sample outputs
            if (step % 10 == 0 and step > 0) or (current_loss < 0.1 and current_loss == buffer.get_lowest_loss()):
                all_outputs = []
                prompt_success = []
                
                for i in range(num_prompts):
                    tmp_messages = copy.deepcopy(multi_messages[i])
                    if isinstance(tmp_messages, str):
                        tmp_messages = [{"role": "user", "content": tmp_messages}]
                    
                    for j in range(len(tmp_messages)):
                        if "{optim_str}" in tmp_messages[j]["content"]:
                            tmp_messages[j]["content"] = tmp_messages[j]["content"].replace("{optim_str}", optim_str)
                            break
                    
                    input = tokenizer.apply_chat_template(tmp_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    output = model.generate(input, attention_mask=torch.ones_like(input), do_sample=False, max_new_tokens=20)
                    output_text = tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]
                    all_outputs.append(output_text)
                    
                    # Check success for this prompt
                    individual_success = check_success(output_text, multi_targets[i], early_stop_condition)
                    prompt_success.append(individual_success)

                # Overall success if all prompts succeed
                success = all(prompt_success)
                print(f"Step {step}\n Targets: {multi_targets}\n Outputs: {all_outputs}\n Check success: {prompt_success}")
            
            if config.early_stop and success:
                self.stop_flag = True
                print("Early stopping due to finding a perfect match on all prompts.") 
            
            # if step % 10 == 0 or success:
            #     log_step = {
            #         "step": step, 
            #         "loss": current_loss, 
            #         "success": success,
            #         "individual_success": prompt_success,
            #         "optim_str": optim_str, 
            #         "outputs": all_outputs, 
            #         "targets": multi_targets
            #     }
            #     f.write(json.dumps(log_step, ensure_ascii=False) + "\n")
            #     f.flush()

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
        attn_str: str,
        target: str,
        early_stop_condition: str,
        attn_method: str = "layer_mean",
        attn_config: dict = None,
    ) -> GCGResult:
        """Run GCG optimization with attention-based loss.
        
        Args:
            messages: The conversation to use for optimization
            attn_str: The string in messages whose attention weights should be minimized
            target: The target generation
            early_stop_condition: The injected task description for success checking
            attn_method: Method for computing attention loss (currently always averages across all layers)
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
    
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
    
        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + " {optim_str}"

        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 


        # if "{optim_str}" in template and " {optim_str}" not in template:
        #     template = template.replace("{optim_str}", " {optim_str}")

        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        target = " " + target if config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Find attention string token indices in the before_str
        attn_str_ids = tokenizer([attn_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        attn_indices = self._find_attn_str_indices(before_ids, attn_str_ids[:, 1:])
        
        if attn_indices is None:
            raise ValueError(f"Attention string '{attn_str}' not found in the input messages")

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

                if config.early_stop and check_success(output, target, early_stop_condition):
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

    def init_buffer_multi(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        print(f"Initializing attack buffer of size {config.buffer_size} for multi-prompt attack...")

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

        # Compute the loss on the initial buffer entries across all prompts
        # Create a wrapper function that matches find_executable_batch_size expectations
        def compute_init_loss_wrapper(batch_size_arg):
            return self.compute_candidates_loss_multi(init_buffer_ids, batch_size_arg)
        
        init_buffer_losses = find_executable_batch_size(compute_init_loss_wrapper, true_buffer_size)()

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer, "", "")

        print("Initialized attack buffer for multi-prompt attack.")
        
        return buffer

    def compute_token_gradient_multi(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix averaged across all prompts.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized 
        """
        model = self.model
        embedding_layer = self.embedding_layer
        num_prompts = len(self.all_target_ids)

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        total_loss = 0
        
        for i in range(num_prompts):
            if self.all_prefix_caches[i]:
                input_embeds = torch.cat([optim_embeds, self.all_after_embeds[i], self.all_target_embeds[i]], dim=1)
                output = model(inputs_embeds=input_embeds, past_key_values=self.all_prefix_caches[i], use_cache=True)
            else:
                input_embeds = torch.cat([self.all_before_embeds[i], optim_embeds, self.all_after_embeds[i], self.all_target_embeds[i]], dim=1)
                output = model(inputs_embeds=input_embeds)

            logits = output.logits

            # Shift logits so token n-1 predicts token n
            shift = input_embeds.shape[1] - self.all_target_ids[i].shape[1]
            shift_logits = logits[..., shift-1:-1, :].contiguous() # (1, num_target_ids, vocab_size)
            shift_labels = self.all_target_ids[i]

            if self.config.use_mellowmax:
                label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
            else:
                loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss += loss

        # Average the loss across all prompts
        avg_loss = total_loss / num_prompts
        optim_ids_onehot_grad = torch.autograd.grad(outputs=[avg_loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def compute_candidates_loss_multi(
        self,
        sampled_ids: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences across all prompts.

        Args:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                the candidate sequences to evaluate
            batch_size : int
                the number of candidate sequences to evaluate in a given batch
        """
        all_loss = []
        num_prompts = len(self.all_target_ids)
        embedding_layer = self.embedding_layer

        for i in range(0, sampled_ids.shape[0], batch_size):
            with torch.no_grad():
                sampled_ids_batch = sampled_ids[i:i+batch_size]
                current_batch_size = sampled_ids_batch.shape[0]
                
                # Initialize loss accumulator for this batch
                batch_total_loss = torch.zeros(current_batch_size, device=self.model.device)
                
                # Compute loss for each prompt
                for prompt_idx in range(num_prompts):
                    if self.all_prefix_caches[prompt_idx]:
                        input_embeds = torch.cat([
                            embedding_layer(sampled_ids_batch),
                            self.all_after_embeds[prompt_idx].repeat(current_batch_size, 1, 1),
                            self.all_target_embeds[prompt_idx].repeat(current_batch_size, 1, 1),
                        ], dim=1)
                        
                        prefix_cache_batch = []
                        if not prefix_cache_batch or current_batch_size != batch_size:
                            prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.all_prefix_caches[prompt_idx][j]] for j in range(len(self.all_prefix_caches[prompt_idx]))]

                        outputs = self.model(inputs_embeds=input_embeds, past_key_values=prefix_cache_batch, use_cache=True)
                    else:
                        input_embeds = torch.cat([
                            self.all_before_embeds[prompt_idx].repeat(current_batch_size, 1, 1),
                            embedding_layer(sampled_ids_batch),
                            self.all_after_embeds[prompt_idx].repeat(current_batch_size, 1, 1),
                            self.all_target_embeds[prompt_idx].repeat(current_batch_size, 1, 1),
                        ], dim=1)
                        outputs = self.model(inputs_embeds=input_embeds)

                    logits = outputs.logits

                    tmp = input_embeds.shape[1] - self.all_target_ids[prompt_idx].shape[1]
                    shift_logits = logits[..., tmp-1:-1, :].contiguous()
                    shift_labels = self.all_target_ids[prompt_idx].repeat(current_batch_size, 1)
                    
                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else:
                        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

                    loss = loss.view(current_batch_size, -1).mean(dim=-1)
                    batch_total_loss += loss

                    # if self.config.early_stop:
                    #     if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                    #         self.stop_flag = True

                    del outputs
                    gc.collect()
                    torch.cuda.empty_cache()

                # Average loss across all prompts for this batch
                avg_batch_loss = batch_total_loss / num_prompts
                all_loss.append(avg_batch_loss)

        return torch.cat(all_loss, dim=0)
    
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

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, use_cache=True, output_hidden_states=True)
        else:
            input_embeds = torch.cat([self.before_embeds, optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, output_hidden_states=True)

        logits = output.logits
        hidden_states = output.hidden_states

        # Compute original GCG loss
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift-1:-1, :].contiguous()
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            gcg_loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            gcg_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Compute attention loss
        attn_loss = self._compute_attention_loss(hidden_states, input_embeds.shape[1])
        
        # Combine losses
        beta = self.attn_config.get("beta", 1.0)
        total_loss = gcg_loss + beta * attn_loss

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[total_loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def _compute_attention_loss(self, hidden_states: tuple, seq_len: int) -> Tensor:
        """Compute attention loss for the first output token attending to the attention string.
        Averages attention weights across all layers.
        
        Args:
            hidden_states: Hidden states from all layers
            seq_len: Length of the input sequence
            
        Returns:
            Attention loss scalar (averaged across all layers)
        """
        from gcg_src.attention_utils import get_attention_weights_one_layer, infer_model_type
        
        model = self.model
        model_type = infer_model_type(model)
        
        # Get number of layers
        try:
            num_layers = len(model.model.layers)
        except:
            num_layers = len(model.model.language_model.layers)
        
        layer_attention_losses = []
        
        # Compute attention loss for each layer
        for layer_idx in range(num_layers):
            # Get attention weights for this layer
            # We want attention from the first output token (position seq_len) to the attention string
            attn_weights = get_attention_weights_one_layer(
                model, 
                hidden_states, 
                layer_idx,
                attribution_start=seq_len,  # First output token
                attribution_end=seq_len + 1,  # Only first output token
                model_type=model_type
            )
            
            # attn_weights shape: [num_heads, 1, seq_len-1] (for first output token attending to all input tokens)
            # We want to minimize attention to the attention string tokens
            attn_to_str = attn_weights[:, :, :, self.attn_indices].mean()  # Average over heads and attention string tokens
            layer_attention_losses.append(attn_to_str)
    
        avg_attention_loss = torch.stack(layer_attention_losses).mean()


        # attn_weights = get_attention_weights_one_layer(
        #     model, 
        #     hidden_states, 
        #     10,
        #     attribution_start=seq_len,  # First output token
        #     attribution_end=seq_len + 1,  # Only first output token
        #     model_type=model_type
        # )
        
        # # attn_weights shape: [num_heads, 1, seq_len-1] (for first output token attending to all input tokens)
        # # We want to minimize attention to the attention string tokens
        # attn_to_str = attn_weights[:, :, :, self.attn_indices].mean() 
        # avg_attention_loss = attn_to_str

        
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

                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids_batch),
                        self.after_embeds.repeat(current_batch_size, 1, 1),
                        self.target_embeds.repeat(current_batch_size, 1, 1),
                    ], dim=1)
                    
                    prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[j]] for j in range(len(self.prefix_cache))]
                    outputs = self.model(inputs_embeds=input_embeds, past_key_values=prefix_cache_batch, use_cache=True, output_hidden_states=True)
                else:
                    input_embeds = torch.cat([
                        self.before_embeds.repeat(current_batch_size, 1, 1),
                        embedding_layer(sampled_ids_batch),
                        self.after_embeds.repeat(current_batch_size, 1, 1),
                        self.target_embeds.repeat(current_batch_size, 1, 1),
                    ], dim=1)
                    outputs = self.model(inputs_embeds=input_embeds, output_hidden_states=True)

                logits = outputs.logits
                hidden_states = outputs.hidden_states

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

                # Compute attention loss for each sample in the batch
                attn_losses = []
                for b in range(current_batch_size):
                    # Extract hidden states for this batch item
                    batch_hidden_states = tuple(h[b:b+1] for h in hidden_states)
                    attn_loss = self._compute_attention_loss(batch_hidden_states, input_embeds.shape[1])
                    attn_losses.append(attn_loss)
                
                attn_losses = torch.stack(attn_losses)
                
                # Combine losses
                beta = self.attn_config.get("beta", 1.0)
                total_loss = gcg_loss + beta * attn_losses
                # avg_gcg_loss = gcg_loss.mean().item()
                # avg_attn_loss = attn_losses.mean().item()

                print(f"GCG loss: {gcg_loss.mean().item()}, Attention loss: {attn_losses.mean().item()}, Total loss: {total_loss.mean().item()}")
                
                all_loss.append(total_loss)

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def run_attn_only(
        self,
        messages: Union[str, List[dict]],
        attn_str: str,
        attn_config: dict = None,
    ) -> GCGResult:
        """Run GCG optimization with only attention-based loss (no target generation loss).
        
        Args:
            messages: The conversation to use for optimization
            attn_str: The string in messages whose attention weights should be minimized
            early_stop_condition: The injected task description for success checking
            attn_method: Method for computing attention loss (currently always averages across all layers)
            attn_config: Configuration for attention loss (beta weight for scaling)
        
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

        # Tokenize everything that doesn't get optimized
        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        # Find attention string token indices in the before_str
        attn_str_ids = tokenizer([attn_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
        attn_indices = self._find_attn_str_indices(before_ids, attn_str_ids[:, 1:])
        
        if attn_indices is None:
            raise ValueError(f"Attention string '{attn_str}' not found in the input messages")

        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        before_embeds, after_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids)]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values
        
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.attn_indices = attn_indices
        self.attn_config = attn_config

        # Initialize the attack buffer (modified for attention-only)
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

                # Compute loss on all candidate sequences with attention loss only
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                loss = find_executable_batch_size(self.compute_candidates_loss_attn_only, batch_size)(sampled_ids)
                
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            # For attention-only optimization, we can't use target-based success checking
            # Instead, we can check if attention loss is sufficiently low
            nice_print(f"Step {step}: Attention loss = {current_loss:.6f}, Optim string: '{optim_str}'")
            if config.early_stop and (step % 10 == 0 or float(current_loss) < 0.01):
                
                # Simple success criterion: very low attention loss
                if config.early_stop and current_loss < 0.001:
                    print("Early stopping due to very low attention loss.") 
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

    def compute_token_gradient_attn_only(self, optim_ids: Tensor) -> Tensor:
        """Computes the gradient of only the attention loss w.r.t the one-hot token matrix.

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
            input_embeds = torch.cat([optim_embeds, self.after_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, use_cache=True, output_hidden_states=True)
        else:
            input_embeds = torch.cat([self.before_embeds, optim_embeds, self.after_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, output_hidden_states=True)

        hidden_states = output.hidden_states

        # Compute only attention loss
        attn_loss = self._compute_attention_loss(hidden_states, input_embeds.shape[1])
        
        # Scale by beta if provided
        beta = self.attn_config.get("beta", 1.0)
        total_loss = beta * attn_loss

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[total_loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def compute_candidates_loss_attn_only(self, batch_size: int, sampled_ids: Tensor) -> Tensor:
        """Computes only the attention loss on all candidate token id sequences.

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

                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids_batch),
                        self.after_embeds.repeat(current_batch_size, 1, 1),
                    ], dim=1)
                    
                    prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[j]] for j in range(len(self.prefix_cache))]
                    outputs = self.model(inputs_embeds=input_embeds, past_key_values=prefix_cache_batch, use_cache=True, output_hidden_states=True)
                else:
                    input_embeds = torch.cat([
                        self.before_embeds.repeat(current_batch_size, 1, 1),
                        embedding_layer(sampled_ids_batch),
                        self.after_embeds.repeat(current_batch_size, 1, 1),
                    ], dim=1)
                    outputs = self.model(inputs_embeds=input_embeds, output_hidden_states=True)

                hidden_states = outputs.hidden_states

                # Compute attention loss for each sample in the batch
                attn_losses = []
                for b in range(current_batch_size):
                    # Extract hidden states for this batch item
                    batch_hidden_states = tuple(h[b:b+1] for h in hidden_states)
                    attn_loss = self._compute_attention_loss(batch_hidden_states, input_embeds.shape[1])
                    attn_losses.append(attn_loss)
                
                attn_losses = torch.stack(attn_losses)
                
                # Scale by beta if provided
                beta = self.attn_config.get("beta", 1.0)
                total_loss = beta * attn_losses

                # print(f"Attention loss: {attn_losses.mean().item()}, Total loss: {total_loss.mean().item()}")
                
                all_loss.append(total_loss)

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def init_buffer_attn_only(self) -> AttackBuffer:
        """Initialize attack buffer for attention-only optimization."""
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        print(f"Initializing attack buffer of size {config.buffer_size} for attention-only optimization...")

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
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = find_executable_batch_size(self.compute_candidates_loss_attn_only, true_buffer_size)(init_buffer_ids)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer, "", "")

        print("Initialized attack buffer for attention-only optimization.")
        
        return buffer

# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    early_stop_condition: str,
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
    result = gcg.run(messages, target, early_stop_condition)
    # gc.collect()
    # torch.cuda.empty_cache()
    return result

# A wrapper around the GCG `run_multi` method that provides a simple API
def run_multi(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    multi_messages: List[Union[str, List[dict]]],
    multi_targets: List[str],
    early_stop_condition: str = 'start_with',
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG that works across multiple prompts. 

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        multi_messages: List of conversations to use for optimization.
        multi_targets: List of target generations corresponding to each conversation.
        early_stop_condition: The injected task description for success checking.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    # logger.setLevel(getattr(logging, config.verbosity))
    
    gcg = GCG(model, tokenizer, config)
    result = gcg.run_multi(multi_messages, multi_targets, early_stop_condition)
    # gc.collect()
    # torch.cuda.empty_cache()
    return result

# A wrapper around the GCG `run_attn` method that provides a simple API
def run_attn(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    attn_str: str,
    target: str,
    early_stop_condition: str,
    attn_method: str = "layer_mean",
    attn_config: Optional[dict] = None,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG with attention-based loss.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        attn_str: The string in messages whose attention weights should be minimized.
        target: The target generation.
        early_stop_condition: The injected task description for success checking.
        attn_method: Method for computing attention loss (currently always averages across all layers).
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
    result = gcg.run_attn(messages, attn_str, target, early_stop_condition, attn_method, attn_config)
    return result

# A wrapper around the GCG `run_attn_only` method that provides a simple API
def run_attn_only(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    attn_str: str,
    attn_config: Optional[dict] = None,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG with only attention-based loss.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        attn_str: The string in messages whose attention weights should be minimized.
        early_stop_condition: The injected task description for success checking.
        attn_config: Configuration for attention loss (beta weight for scaling).
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    if attn_config is None:
        attn_config = {"beta": 1.0}
    
    gcg = GCG(model, tokenizer, config)
    result = gcg.run_attn_only(messages, attn_str, attn_config)
    return result
    