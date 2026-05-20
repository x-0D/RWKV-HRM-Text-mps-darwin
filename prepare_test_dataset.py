"""
Generate a small synthetic test dataset for HRM-Text (RWKV) training on MPS/GPU.

Produces the same on-disk format as the data_io pipeline:
  {output_path}/
    metadata.json
    tokens.npy          ->  uint16, shape (total_tokens,)
    epoch_0/
      inst_start.npy    ->  int32, shape (num_samples,)
      inst_len.npy      ->  int32, shape (num_samples,)
      resp_start.npy    ->  int32, shape (num_samples,)
      resp_len.npy      ->  int32, shape (num_samples,)

Usage:
    python prepare_test_dataset.py
    python pretrain.py arch/net@arch=hrm_rwkv arch/size@arch=rwkv_tiny \
        data.path=test_dataset lr=3e-4 global_batch_size=1024 \
        fwd_bwd_dtype=bfloat16 epochs=2
"""

import json
import os
import numpy as np


def generate_test_dataset(
    output_path: str = "test_dataset",
    num_samples: int = 500,
    max_seq_len: int = 1024,
    vocab_size: int = 65536,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)

    os.makedirs(os.path.join(output_path, "epoch_0"), exist_ok=True)

    # Generate synthetic samples: random integers for instructions and responses
    samples = []
    for _ in range(num_samples):
        inst_len = rng.randint(8, 128)
        resp_len = rng.randint(8, min(128, max_seq_len - inst_len))
        inst = rng.randint(5, vocab_size, size=inst_len, dtype=np.int32)
        resp = rng.randint(5, vocab_size, size=resp_len, dtype=np.int32)
        samples.append((inst, resp))

    # Concatenate into one big token array
    all_tokens = []
    inst_start = []
    inst_len_arr = []
    resp_start = []
    resp_len_arr = []

    offset = 0
    for inst, resp in samples:
        inst_start.append(offset)
        inst_len_arr.append(len(inst))
        all_tokens.extend(inst.tolist())
        offset += len(inst)

        resp_start.append(offset)
        resp_len_arr.append(len(resp))
        all_tokens.extend(resp.tolist())
        offset += len(resp)

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    inst_start = np.array(inst_start, dtype=np.int32)
    inst_len_arr = np.array(inst_len_arr, dtype=np.int32)
    resp_start = np.array(resp_start, dtype=np.int32)
    resp_len_arr = np.array(resp_len_arr, dtype=np.int32)

    # Save tokens
    np.save(os.path.join(output_path, "tokens.npy"), all_tokens)

    # Save epoch index arrays
    epoch_dir = os.path.join(output_path, "epoch_0")
    np.save(os.path.join(epoch_dir, "inst_start.npy"), inst_start)
    np.save(os.path.join(epoch_dir, "inst_len.npy"), inst_len_arr)
    np.save(os.path.join(epoch_dir, "resp_start.npy"), resp_start)
    np.save(os.path.join(epoch_dir, "resp_len.npy"), resp_len_arr)

    # metadata (max_seq_len accounts for +1 AR shift)
    metadata = {
        "tokenizer_info": {"vocab_size": vocab_size},
        "max_seq_len": max_seq_len + 1,
        "total_length": len(all_tokens),
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Test dataset created at '{output_path}':")
    print(f"  Samples:       {num_samples}")
    print(f"  Total tokens:  {len(all_tokens)}")
    print(f"  Vocab size:    {vocab_size}")
    print(f"  Max seq len:   {max_seq_len}")
    print(f"  Token dtype:   {all_tokens.dtype}")
    print()

    # Quick sanity
    print("Files:")
    for fname in ["tokens.npy", "epoch_0/inst_start.npy", "epoch_0/inst_len.npy",
                   "epoch_0/resp_start.npy", "epoch_0/resp_len.npy"]:
        path = os.path.join(output_path, fname)
        arr = np.load(path, mmap_mode="r")
        print(f"  {fname}: shape={arr.shape} dtype={arr.dtype}")
    print()
    print("Done.")


if __name__ == "__main__":
    generate_test_dataset()
