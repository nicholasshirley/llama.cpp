# DSA query chunking for Kimi-K3.1 (an experiment)

This branch runs the DSA attention block of Kimi-K3.1 in query chunks when the ubatch is large. On one 72 GB card the
context window grows from 262k to 491k tokens at ub 8192, and prefill and decode speed stay the same.

It is a reference for the discussion in #30815, not a pull request. An AI coding agent (model claude-opus-5-5) wrote
the code and this note under the repo owner's direction, and ran every measurement below on the owner's machine.

## The problem

DSA (DeepSeek sparse attention, as Kimi-K3.1 uses it) picks the 2048 cached tokens that each new token attends to. A
"lightning indexer" scores every cached token for every query in the ubatch, `top_k` picks the best 2048, and a top-k
mask turns the picks into an additive mask for flash attention. On CUDA each DSA layer holds three buffers of size
[n_kv × n_ubatch]:

| Buffer | Type | Size at 262,144 cells × ub 8192 |
|---|---|---|
| KQ mask (a host input, copied to the device) | f16 | 4 GiB |
| indexer score | f32 | 8 GiB |
| top-k mask | f16 | 4 GiB |

The score and the top-k mask do not live at the same time, so the peak is the mask plus the score: 6 bytes per
(cell, query), about 48 KiB per context token at ub 8192. The KV cache costs only 17.5 KiB per token (q8_0 MLA latent
plus the indexer keys), so the window was bound by this scratch, not by the cache.

A small ubatch would shrink the scratch, but this setup needs a large one. The experts live in host RAM (`-cmoe`), and
each prefill pass uploads them to the GPU, which takes about 29 s per pass. A big ubatch spreads that cost over more
tokens: on one card a 32k-token prefill ran at 141 t/s with ub 8192, 94 t/s with ub 4096 and 56 t/s with ub 2048
(the build before this branch; master is faster, see below).

## The change

`LLAMA_DSA_CHUNK=N` makes each DSA layer process a single-stream ubatch N queries at a time (unset or 0 keeps the old
graph). For each layer:

1. The indexer computes the score of one chunk of queries, and `top_k` turns it into the top-k cells of that chunk.
   The graph orders these as score₀, top-k₀, score₁, top-k₁, …, so ggml-alloc frees each score before it allocates
   the next one, and all chunks share one buffer.
2. For each chunk, the top-k mask and flash attention run on that chunk's queries, and the outputs are concatenated.
   Each chunk's top-k mask is freed after its flash attention.

A query's selection does not depend on the other queries in the ubatch, so the math is unchanged. The code is all in
`src/models/kimi-k31.cpp` and `src/models/models.h`; no kernel or ggml op changes.

**The footgun that this hit:** the KQ mask is a host input, and the scheduler copies each view tensor of a host input
to the device on its own, as a separate split input that is allocated at the start of its split. The first version
made new views of the mask in every layer, so each layer allocated eight device copies at once (twice the size of the
mask) and the saving almost vanished. The branch builds the chunk views once per graph and shares them across all
DSA layers, so the device copies of the views add up to one mask, as before. qwen4exp in #30815 already does this:
`build_inp_kpool` makes its row views once.

## Results

The machine has two RTX PRO 5000 72 GB Blackwell cards, an EPYC 9355P and 1.5 TB of RAM. The model is Kimi-K3.1-Pro
with the experts in MXFP4 and the rest in Q5_K or q8_0 (1.4 TB), run with `-cmoe -fa on -ctk q8_0 -ctv q8_0 -np 1`
and 8 active experts instead of 16 (`--override-kv kimi-k31.expert_used_count=int:8`). Chunk size is 1024.

### Compute buffer (llama-fit-params projection, CUDA0, MiB)

| Layout | Context | Unchunked | Chunked |
|---|---|---|---|
| one card, ub 8192 | 262,144 | 20,902 | 15,980 |
| one card, ub 8192 | 524,288 | 33,190 | 20,076 |
| one card, ub 8192 | 1,048,576 | 57,766 | 28,268 |
| two cards (`-sm layer -ts 1,1`), ub 16384 | 655,360 | 73,670 / 77,670 | 36,044 / 36,710 |

After chunking, the compute buffer grows by 16 KiB per context token at ub 8192, which is the f16 KQ mask alone. The
peak then sits in an MoE split: the 5 GB device copy of one expert tensor (`-cmoe` op offload), the mask copies and,
at 1M context, a 4.5 GB free block that is too small for the next expert copy.

### Measured runs

The probe is a 32,600-token prefill from an empty cache, then 128 decoded tokens. Speeds come from the server's own
timings.

| Run | Layout | Context | ub | Chunk | Prefill t/s | Decode t/s at 33k | Peak VRAM (MiB) |
|---|---|---|---|---|---|---|---|
| D0 | one card | 262,144 | 8192 | off | 153.0 | 10.02 | 67,164 |
| D1 | one card | 262,144 | 8192 | 1024 | 154.0 | 10.08 | 62,242 |
| D2 | one card | 491,520 | 8192 | 1024 | 152.9 | 10.07 | 69,752 |
| E1 | two cards | 786,432 | 16384 | 1024 | 207.9 | 10.05 | 68,004 / 68,772 |

- Chunking costs no measurable speed (D1 against D0).
- The live peak matched the llama-fit-params projection plus 950 MiB to the MiB in D0, D1 and D2, so the projection
  can size a window without loading the model.
- A needle test on E1 found 4 of 4 anchors in a 323,123-token document.

## Correctness

**The tiny model:** `test-llama-archs -a kimi-k31 -o DIR` writes a random 20 MB K3.1. With `llama-perplexity -c 256
-b 256 -ub 256 --chunks 8 --kl-divergence-base`, the chunked runs match the unchunked run bit for bit (KLD 0, all top
tokens equal): chunk sizes 64, 128 and 255 on CPU, and 64 and 128 on CUDA, also with `-cmoe` and a q8_0 cache.
`test-llama-archs -a kimi-k31` also passes (CPU against CUDA) with `LLAMA_DSA_CHUNK=7` and `=16`. That model has a
single DSA layer and a top-k larger than its cache, so it does not test cross-layer sharing or a real selection.

**The real model:** Kimi-K3.1 moves its output distribution by a KL divergence of about 0.03-0.04 at ctx 2048 (and
0.074 at ctx 32768) under any change to the batch shape. Turning flash attention off, changing the ubatch from 2048 to
512 and splitting flash attention into 512-query calls all land in that range. An identical rerun reproduces to KLD 0
(maximum 5e-5). So "chunked against unchunked" cannot reach the kernel noise floor, and each check here compares the
chunked run with an unchunked run at a smaller ubatch.

| wikitext-2, ctx 32768, 2 chunks | PPL | KLD against the base | Same top token |
|---|---|---|---|
| base: unchunked, ub 8192 | 1.4748 | | |
| chunked 1024, ub 8192 | 1.4765 | 0.0736 | 94.36 % |
| unchunked, ub 4096 | 1.4798 | 0.0738 | 94.39 % |

The chunked run moves the output less than an ordinary ubatch change does.

## What is left

- **The dense KQ mask** is now the only scratch that grows with context × ubatch: 2 bytes per (cell, query), and on
  every card that holds a DSA layer. If each chunk built its own mask on the device from the cell positions and the
  query positions, the host would never fill or upload the full mask. A rough estimate (nothing built yet) puts one
  card near the model's 1M window at ub 8192 then, because the KV cache would be the only part left that grows with
  context.
- **Flash attention from the selected indices** (the idea raised in #30815) would remove the per-chunk top-k mask.
  After chunking that buffer is only 2 bytes per (cell, chunk query), so the gain would be speed more than memory.
- **The other DSA models** (`glm-dsa.cpp`, `deepseek32.cpp`) use the same indexer, top-k and top-k mask path, so the
  change should port directly. Only Kimi-K3.1 was measured.
- **Multi-stream ubatches** (several sequences with a split cache) are not chunked.

## Reproduce

```sh
cmake -B build -DGGML_CUDA=ON
cmake --build build -j --target llama-server llama-fit-params test-llama-archs llama-perplexity

# the projection: llama-fit-params reads the GGUF metadata and tensor shapes, not the weights
LLAMA_DSA_CHUNK=1024 build/bin/llama-fit-params -m Kimi-K3.1-Pro-00001-of-000NN.gguf -c 491520 -b 8192 -ub 8192 \
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -cmoe -np 1 -fitp on

# serving
LLAMA_DSA_CHUNK=1024 build/bin/llama-server -m ... -c 491520 -b 8192 -ub 8192 \
    -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -cmoe -np 1

# the tiny-model check
build/bin/test-llama-archs -a kimi-k31 -o /tmp/k31
build/bin/llama-perplexity -m /tmp/k31/kimi-k31-moe.gguf -f wiki.test.raw -c 256 -b 256 -ub 256 --chunks 8 \
    --kl-divergence-base /tmp/k31/base.kld
LLAMA_DSA_CHUNK=64 build/bin/llama-perplexity -m /tmp/k31/kimi-k31-moe.gguf -f wiki.test.raw -c 256 -b 256 -ub 256 \
    --chunks 8 --kl-divergence-base /tmp/k31/base.kld --kl-divergence
```
