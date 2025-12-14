import os
import json, math, random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
import re
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
import peft

def clean_text(t: str) -> str:
    '''
    Cleans the text, returns empty if not string and replaces line breaks and multiple spaces with a single space. 
    '''
    if not isinstance(t, str):
        return ""
    t = t.replace("\n", " ").replace("\t", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def make_context_continuation_from_texts(
    texts: List[str], tokenizer=None, ctx_token_limit=150
):
    '''
	Docstring for make_context_continuation_from_texts
	
	:param texts: Description
	:type texts: List[str]
	:param tokenizer: Description
	:param ctx_token_limit: Description
	'''
    pairs = {"context": [], "continuation": []}

    for doc in texts:
        # Sentence split with backup regex if spaCy/NLTK not used
        sents = re.split(r"(?<=[.!?])\s+", doc)
        sents = [s.strip() for s in sents if len(s.strip()) > 0]

        if len(sents) < 2:
            continue

        ctx = ""
        used = 0

        for i, s in enumerate(sents[:-1]):
            candidate = (ctx + " " + s).strip()

            # tokenizer-aware token limit
            if tokenizer:
                length = len(tokenizer(candidate)["input_ids"])
            else:
                length = len(candidate.split())

            if length <= ctx_token_limit:
                ctx = candidate
                used = i
            else:
                break

        cont = sents[used + 1]
        pairs["context"].append(ctx)
        pairs["continuation"].append(cont)

    return pairs


# -------- A. ASAP Essays (Kaggle → HuggingFace mirror) --------
def load_asap_essays():
    ds = load_dataset("llm-aes/asap-8-original")  # community mirror
    essays = [clean_text(e["essay"]) for e in ds["train"]]
    return essays


# -------- B. IELTS/TOEFL Essays (Open Source Mirror) --------
def load_toefl_essays():
    ds = load_dataset("text", data_files="https://huggingface.co/datasets/hellonlp/toefl-essays/resolve/main/toefl_essays.txt")
    essays = [clean_text(x["text"]) for x in ds["train"]]
    return essays


# -------- C. Wikipedia (English) --------
def load_wikipedia(limit=5000):
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", stream = True)
    texts = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        texts.append(clean_text(row["text"]))
    return texts


# -------- D. arXiv abstracts --------
def load_arxiv(limit=5000):
    ds = load_dataset("arxiv", split="train")
    texts = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        texts.append(clean_text(row["abstract"]))
    return texts


# -------- E. S2ORC subset (Open academic papers) --------
def load_s2orc(limit=5000):
    ds = load_dataset("allenai/s2orc", "comm_use_subset", split="train", streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        if "abstract" in row and row["abstract"]:
            texts.append(clean_text(row["abstract"]))
    return texts

def build_academic_dataset(tokenizer=None, limit_each=2000):
    print("Loading datasets...")

    all_texts = []

    try:
        print("Loading ASAP Essays")
        all_texts += load_asap_essays()[:limit_each]
    except:
        print("ASAP Essays failed to load.")
    print(f"Total raw documents loaded: {len(all_texts)}")
    
    print("Generating (context, continuation) pairs...")
    pairs = make_context_continuation_from_texts(all_texts, tokenizer)

    df = pd.DataFrame(pairs)
    print(f"Generated {len(df)} training pairs.")

    return df