#!/usr/bin/env python3
"""
Prepare HRM-Text training data from pre-cleaned HuggingFace dataset.
Downloads pre-cleaned data, tokenizes with BPE 65K tokenizer,
and creates epoch structure compatible with V1Dataset.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from tokenizers import Tokenizer

# Special token IDs (from HRM-Text tokenizer vocab)
BOQ_ID = 30    # � (</think>)
EOQ_ID = 27    # ￾ (</think>)
EOA_ID = 28    # ￿ (</think>)
COND_TOKEN_IDS = {"direct": 1, "cot": 2, "noisy": 3, "synth": 4}

SUBSETS = {
    "small": ["gsm8k_train.jsonl", "math_train.jsonl"],
    "medium": ["gsm8k_train.jsonl", "math_train.jsonl", "amps_khan.jsonl",
                 "omnimath.jsonl", "webinstruct_verified.jsonl", "no_robots.jsonl"],
    "full": "ALL",
}

HF_REPO = "sapientinc/HRM-Text-data-io-cleaned-20260515"
TOKENIZER_REL = "../data_io/trained_tokenizers/bpe/tokenizer.json"

def load_tokenizer(tokenizer_path):
    print(f"Loading tokenizer: {tokenizer_path}")
    tok = Tokenizer.from_file(tokenizer_path)
    print(f"  Vocab size: {len(tok.get_vocab())}")
    return tok

def list_data_files(subset, datasets_override=None):
    if datasets_override:
        return [("data/" + f, f) for f in datasets_override]
    if SUBSETS[subset] != "ALL":
        return [("data/" + f, f) for f in SUBSETS[subset]]
    print("Listing all files in HF repo...")
    all_files = list_repo_files(HF_REPO, repo_type="dataset")
    data_files = []
    for f in all_files:
        if f.startswith("data/") and f.endswith(".jsonl"):
            data_files.append((f, os.path.basename(f)))
        elif f.startswith("data_clustered/") and f.endswith(".parquet"):
            data_files.append((f, os.path.basename(f)))
    if subset == "medium":
        jsonl_files = [(f, n) for f, n in data_files if f.endswith(".jsonl")]
        parquet_files = [(f, n) for f, n in data_files if f.endswith(".parquet")][:5]
        return jsonl_files + parquet_files
    return data_files

def download_and_read(repo_path, name):
    # Find cached file directly
    cache_base = os.path.expanduser('~/.cache/huggingface/hub/datasets--sapientinc--HRM-Text-data-io-cleaned-20260515/snapshots')
    snapshots = os.listdir(cache_base) if os.path.exists(cache_base) else []
    if snapshots:
        snapshot = snapshots[0]
        local = os.path.join(cache_base, snapshot, repo_path)
        if os.path.exists(local):
            pass  # Use cached file
        else:
            local = hf_hub_download(repo_id=HF_REPO, filename=repo_path)
    else:
        local = hf_hub_download(repo_id=HF_REPO, filename=repo_path)
    if repo_path.endswith(".jsonl"):
        import orjson
        rows = []
        with open(local, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(orjson.loads(line))
        return rows
    elif repo_path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(local)
        return df.to_dict("records")
    return []

def tokenize_rows(tokenizer, rows, name):
    print(f"Tokenizing {name}: {len(rows)} rows...")
    all_tokens = []
    inst_starts, inst_lens, resp_starts, resp_lens = [], [], [], []
    for i, row in enumerate(rows):
        condition = row.get("condition", "direct")
        instruction = row.get("instruction", "")
        response = row.get("response", "")
        inst_tokens = [BOQ_ID]
        cond_id = COND_TOKEN_IDS.get(condition, 1)
        inst_tokens.append(cond_id)
        inst_tokens.extend(tokenizer.encode(instruction, add_special_tokens=False).ids)
        inst_tokens.append(EOQ_ID)
        resp_tokens = tokenizer.encode(response, add_special_tokens=False).ids
        resp_tokens.append(EOA_ID)
        inst_start = len(all_tokens)
        inst_len = len(inst_tokens)
        resp_start = inst_start + inst_len
        resp_len = len(resp_tokens)
        inst_starts.append(inst_start)
        inst_lens.append(inst_len)
        resp_starts.append(resp_start)
        resp_lens.append(resp_len)
        all_tokens.extend(inst_tokens)
        all_tokens.extend(resp_tokens)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(rows)} rows")
    return {
        "tokens": np.array(all_tokens, dtype=np.uint16),
        "inst_start": np.array(inst_starts, dtype=np.int64),
        "inst_len": np.array(inst_lens, dtype=np.int64),
        "resp_start": np.array(resp_starts, dtype=np.int64),
        "resp_len": np.array(resp_lens, dtype=np.int64),
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare HRM-Text training data")
    parser.add_argument("--subset", default="small", choices=["small", "medium", "full"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output", default="data/hrm_text_train")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--context-size", type=int, default=4097)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    tokenizer_path = script_dir / TOKENIZER_REL
    tokenizer = load_tokenizer(str(tokenizer_path))

    files = list_data_files(args.subset, args.datasets)
    print(f"Downloading {len(files)} files...")

    all_tokens_list = []
    all_inst_start, all_inst_len, all_resp_start, all_resp_len = [], [], [], []
    current_offset = 0

    for repo_path, name in files:
        try:
            rows = download_and_read(repo_path, name)
            if not rows:
                print(f"  Skipping {name} (empty)")
                continue
            result = tokenize_rows(tokenizer, rows, name)
            all_tokens_list.append(result["tokens"])
            all_inst_start.append(result["inst_start"] + current_offset)
            all_inst_len.append(result["inst_len"])
            all_resp_start.append(result["resp_start"] + current_offset)
            all_resp_len.append(result["resp_len"])
            current_offset += len(result["tokens"])
            print(f"  {name}: {len(rows)} rows, {len(result['tokens']):,} tokens")
        except Exception as e:
            import traceback
            print(f"  ERROR {name}: {e}")
            traceback.print_exc()

    if not all_tokens_list:
        print("No data downloaded!")
        sys.exit(1)

    all_tokens = np.concatenate(all_tokens_list)
    combined_inst_start = np.concatenate(all_inst_start)
    combined_inst_len = np.concatenate(all_inst_len)
    combined_resp_start = np.concatenate(all_resp_start)
    combined_resp_len = np.concatenate(all_resp_len)
    total_rows = len(combined_inst_start)
    print(f"\nTotal: {total_rows:,} rows, {len(all_tokens):,} tokens")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {args.output}...")
    np.save(output_path / "tokens.npy", all_tokens)

    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    for epoch in range(args.epochs):
        epoch_dir = output_path / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        perm = rng.permutation(total_rows)
        np.save(epoch_dir / "inst_start.npy", combined_inst_start[perm])
        np.save(epoch_dir / "inst_len.npy", combined_inst_len[perm])
        np.save(epoch_dir / "resp_start.npy", combined_resp_start[perm])
        np.save(epoch_dir / "resp_len.npy", combined_resp_len[perm])
        print(f"  epoch_{epoch}: {total_rows:,} rows")

    metadata = {
        "tokenizer_info": {"vocab_size": 65536},
        "vocab_size": 65536,
        "max_seq_len": args.context_size,
        "total_length": int(len(all_tokens)),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata.json")
    print(f"\nDone! Dataset ready at: {args.output}")
    print(f"  Rows: {total_rows:,}")
    print(f"  Tokens: {len(all_tokens):,}")
    print(f"  Epochs: {args.epochs}")

if __name__ == "__main__":
    main()
