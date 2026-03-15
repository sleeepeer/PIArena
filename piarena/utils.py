import os
import json
import numpy as np
import random
import warnings
import torch
import re
# from pynvml import *
import copy
from transformers import set_seed
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_json(results, file_path="debug.json"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import sklearn
        sklearn.utils.check_random_state(seed)
        sklearn.random.seed(seed)
    except Exception as e:
        pass
    set_seed(seed) # transformers seed

def find_indices(list1: list, list2: list):
    indices = []
    for element in list1:
        try:
            index = list2.index(element)
            indices.append(index)
        except ValueError:
            continue
    return indices

def contexts_to_paragraphs(contexts):
    paragraphs = contexts[0].split('\n\n')
    paragraphs = ['\n\n'+paragraph for paragraph in paragraphs]

    return paragraphs

def contexts_to_segments(contexts):
    segment_size = 100
    context = contexts[0]
    words = context.split(' ')

    # Create a list to hold segments
    segments = []
    
    # Iterate over the words and group them into segments
    for i in range(0, len(words), segment_size):
        # Join a segment of 100 words and add to segments list
        segment = ' '.join(words[i:i + segment_size])+' '
        segments.append(segment)
    
    return segments

def paragraphs_to_sentences(paragraphs):
    all_sentences = []

    # Split the merged string into sentences
    #sentences = sent_tokenize(merged_string)
    for i,paragraph in enumerate(paragraphs):
        sentences = split_into_sentences(paragraph)
        all_sentences.extend(sentences)
    return all_sentences

def contexts_to_sentences(contexts):
    paragraphs = contexts_to_paragraphs(contexts)
    all_sentences = paragraphs_to_sentences(paragraphs)
    return all_sentences

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_phrases(text: str) -> list[str]:
    sentences = split_into_sentences(text)
    phrases = []
    for sent in sentences:
        phrases+=sent.split(',')
    return phrases

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n","<newline>")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    sentences = [s.replace("<newline>", "\n") for s in sentences]
    return sentences

def nice_print(text):
    print_text = copy.deepcopy(text)
    print_text = print_text.replace("\\", "\\\\").replace("\n", "\\n")
    print(print_text)


def inject(clean_data, injected_prompt, inject_position="end", inject_times=1):
    if inject_position == "random":
        all_sentences = contexts_to_sentences([clean_data])
        num_sentences = len(all_sentences)
        random.seed(num_sentences)
        # Generate random positions
        chosen_positions = []
        for i in range(inject_times):
            random_position = random.randint(int(num_sentences*0), num_sentences)
            # Insert the string at the random position
            all_sentences = all_sentences[:random_position] + [injected_prompt] + all_sentences[random_position:]
            chosen_positions.append(random_position)
            num_sentences += 1  # After each insertion, the number of sentences increases
        context = ' '.join(all_sentences)
    elif inject_position == "end":
        context = clean_data + " " + " ".join([injected_prompt] * inject_times)
    elif inject_position == "start":
        context = " ".join([injected_prompt] * inject_times) + " " + clean_data
    else:
        raise ValueError(f"Invalid inject position: {inject_position}")
    return context