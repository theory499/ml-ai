# ML & AI Learning Roadmap

Derived from `compass_artifact_*.md` (16 domains), `ml_llm_topics.csv` (1,324 topics),
and the `deep_research_*` prompts in this repo. Target reader: senior backend
engineer, zero prior ML, aiming at multimodal-LLM-architect depth.

This document has two parts:

1. **Part A — Lifetime Roadmap.** The full path the compass implies, sequenced into
   8 phases with realistic time estimates and per-phase exit criteria.
2. **Part B — 30-Day Sprint.** A day-by-day intensive plan that gets you to
   *operational fluency on the critical path* (math → PyTorch → DL →
   Transformers → LLMs → multimodal → systems), with one end-to-end capstone.

The repo's `deep_research_system_prompt.md` is the **engine**: at any point in
either plan, take the topic name from `ml_llm_topics.csv`, paste it into the
prompt as `<TOPIC>`, and you get a textbook-grade chapter on it. The roadmap
below tells you *which* topic to feed it next, and *in what order*, so the
chapters compose into mastery rather than a pile of disconnected essays.

---

## Part A — Lifetime Roadmap (Multi-Year)

Honest framing: the compass is a 3–5 year curriculum at 15–25 hrs/week. The
phases below sequence its 16 domains into a path where each phase unlocks the
next. Topic counts in parentheses are from the CSV.

### Phase 1 — Mathematical Foundations (8–12 weeks)

**Goal:** read any ML paper's equations without panic.

- **Linear Algebra (20 topics)** — vectors → matrices → eigendecomp → SVD →
  tensors → einsum. Anchor: MIT 18.06 (Strang) + 3Blue1Brown "Essence of LA."
- **Calculus (17 topics)** — multivariable, gradients, Jacobians, Hessians,
  chain rule, **reverse-mode autodiff** (the link to backprop).
- **Probability (23 topics)** — distributions, expectation, Bayes, MLE/MAP,
  CLT, conjugate priors, importance sampling.
- **Statistics (16 topics)** — bias-variance, hypothesis testing,
  cross-validation, bootstrap, GLMs.
- **Information Theory (12 topics)** — entropy, cross-entropy, KL, mutual
  information, ELBO. This is the language of every loss function.
- **Optimization (24 topics)** — convexity, SGD, momentum, Adam/AdamW, Lion,
  second-order, KKT, mirror descent.
- **Numerical Methods (11 topics)** — FP32/FP16/BF16/FP8, conditioning,
  catastrophic cancellation. Critical for distributed training later.
- *Defer to Phase 8 (research-only):* Measure theory, functional analysis,
  stochastic calculus, topology/differential geometry.

**Exit criterion:** derive backprop by hand for a 2-layer MLP. Derive the
softmax-cross-entropy gradient. Implement SVD-based PCA from scratch.

### Phase 2 — Programming & Tools (3–4 weeks, in parallel with Phase 1)

**Goal:** stop fighting the tools.

- Python advanced (generators, decorators, async, typing).
- **NumPy** (vectorization, broadcasting, einsum). **Pandas** for data.
- **PyTorch** (autograd, `nn.Module`, `DataLoader`, `optim`, `.compile()`,
  distributed primitives) — this is your daily driver.
- **einops** + **Hugging Face stack** (transformers, datasets, tokenizers,
  accelerate, PEFT, TRL).
- Git, W&B or MLflow, Hydra/OmegaConf, pytest, Docker basics.
- *Defer:* JAX/Flax, Triton kernels, CUDA C++, Ray, Spark, Kubernetes — these
  arrive in Phase 7.

**Exit criterion:** implement a CNN in PyTorch, train it on CIFAR-10 with
proper data loading, logging, checkpointing, and reproducible seeds.

### Phase 3 — Classical Machine Learning (5–7 weeks)

**Goal:** internalize the bias-variance / generalization mental model and the
non-deep baselines that every paper compares against.

- ML foundations: empirical risk minimization, PAC, VC dimension, no-free-lunch.
- Supervised: linear/logistic regression, GLMs, decision trees, random forests,
  **gradient boosting (XGBoost/LightGBM)**, SVMs/kernels, Gaussian processes,
  k-NN, naive Bayes.
- Unsupervised: k-means, GMMs (EM), DBSCAN, spectral clustering;
  PCA/kernel-PCA, t-SNE, UMAP; matrix factorization.
- Probabilistic graphical models (Bayes nets, MRFs, CRFs, variable elimination,
  variational inference).
- **Reinforcement Learning core**: MDPs, Bellman, value/policy iteration, TD,
  Q-learning, policy gradients, PPO. (Modern RLHF/GRPO live here conceptually.)
- Other paradigms: meta-learning, few-shot, transfer, active, continual,
  federated.

**Exit criterion:** on a tabular dataset, beat a gradient-boosted baseline
with thoughtful feature engineering and explain *why* via bias-variance.

### Phase 4 — Deep Learning Fundamentals (6–8 weeks)

**Goal:** build any non-transformer architecture from scratch.

- MLPs, universal approximation, activations (ReLU → GELU/SwiGLU), losses
  (CE/MSE/contrastive/CTC/focal).
- Backprop + computational graphs (you derived this in Phase 1; now implement it).
- Initialization (Xavier, He), normalization (BatchNorm → LayerNorm/RMSNorm),
  regularization (dropout, weight decay, label smoothing, mixup).
- Optimizers in practice (Adam family, Lion, Sophia, schedules: cosine, warmup).
- **CNNs** — from LeNet through ResNet/DenseNet → EfficientNet → ConvNeXt.
  Train one on ImageNet-100 or CIFAR.
- **RNNs / LSTM / GRU / seq2seq / attention (Bahdanau, Luong)** — historical
  but essential to understand transformers properly.
- Autoencoders, VAEs, VQ-VAE.
- **Generative models survey:** GANs (DCGAN → StyleGAN → WGAN-GP), normalizing
  flows, autoregressive (PixelCNN), **diffusion (DDPM → DDIM → score-based →
  latent diffusion → classifier-free guidance)**, **flow matching / rectified
  flow** (current frontier).
- Theory: lottery ticket, double descent, NTK, information bottleneck.

**Exit criterion:** implement a ResNet-18 from scratch (not torchvision); train
a small VAE and a small DDPM on MNIST; explain why diffusion replaced GANs.

### Phase 5 — Transformers & LLMs (10–14 weeks; the heart of the curriculum)

**Goal:** be able to read any LLM paper and implement the key idea.

- **Transformer block from scratch** — scaled dot-product attention, MHA, then
  MQA/GQA/MLA, cross-attention, causal masking. Build it in <300 lines.
- **Positional encodings** — sinusoidal → learned → relative → **RoPE** →
  ALiBi → YaRN/NTK-aware (long-context).
- **Tokenization** — BPE, byte-BPE, WordPiece, SentencePiece, Unigram, tiktoken;
  train a tokenizer.
- **Architecture families** — encoder-only (BERT/RoBERTa/DeBERTa), decoder-only
  (GPT, LLaMA 1–4, Mistral, Qwen, DeepSeek, Gemma, Phi), encoder-decoder
  (T5, BART). **SSMs (Mamba, Mamba-2), linear attention (RWKV, RetNet),
  hybrids (Jamba, Granite 4)**, **MoE (Mixtral, DeepSeek-MoE, Switch)**.
- **Pretraining objectives** — CLM, MLM, span corruption, FIM, prefix-LM,
  permutation, ELECTRA-style.
- **Scaling laws** — Kaplan → Chinchilla → compute-optimal → μP →
  inference-time scaling.
- **Fine-tuning** — full SFT, instruction tuning, **PEFT (LoRA/QLoRA/DoRA,
  prefix/prompt tuning, IA³, GaLore)**, continued pretraining, distillation,
  model merging (TIES, DARE, SLERP).
- **Alignment** — RLHF pipeline, reward modeling, PPO, **DPO/IPO/KTO/ORPO/
  SimPO**, **GRPO/DAPO/RLVR (verifiable rewards, 2024–2026 frontier)**,
  RLAIF, Constitutional AI.
- **Reasoning at inference** — CoT, self-consistency, ToT/GoT, ReAct,
  Reflexion, process reward models, **long-CoT models (o1, R1, QwQ)**, MCTS+LLM.
- **RAG** — dense/sparse/hybrid retrieval, rerankers, vector DBs (FAISS,
  Milvus, Qdrant, pgvector), ANN (HNSW, IVF, PQ), ColBERT/late interaction,
  GraphRAG, agentic RAG.
- **Agents & tool use** — function calling, **MCP**, ReAct agents, multi-agent,
  A2A, code/web/computer-use agents, memory, planning.
- **Long context** — sliding window, Longformer/BigBird sparse patterns,
  **ring attention**, **FlashAttention 1/2/3**, FlashInfer, KV cache
  management, StreamingLLM, H2O.
- **Inference optimization** — quantization (INT4/8, GPTQ, AWQ, GGUF,
  SmoothQuant, **FP8 E4M3/E5M2, MXFP8/MXFP4/NVFP4**), KV caching,
  **PagedAttention (vLLM)**, continuous batching, **speculative decoding
  (EAGLE, Medusa, draft models)**, pruning, distillation. Engines: vLLM,
  SGLang, TensorRT-LLM, TGI, llama.cpp.
- **Mechanistic interpretability** — probes, activation patching, logit lens,
  induction heads, **sparse autoencoders (SAEs)**, transcoders, activation
  steering.

**Exit criterion:** implement nanoGPT-style decoder-only transformer from
scratch, pretrain on TinyStories, fine-tune with LoRA, serve with vLLM,
quantize to INT4, and run a basic mech-interp probe on it.

### Phase 6 — Modalities: NLP, Vision, Audio, Multimodal (8–12 weeks)

**Goal:** speak each modality's native language; then fuse them.

- **NLP (66 topics)** — preprocessing, linguistic foundations, classical
  representations (TF-IDF → Word2Vec → GloVe → ELMo → BERT-style →
  Sentence-BERT/SimCSE), core tasks (NER, parsing, coref, NLI, QA, MT,
  summarization), decoding (greedy/beam/top-k/top-p/min-p/contrastive).
- **Computer Vision (94 topics)** — image processing → classical features
  (SIFT, HOG) → CNNs (you did these) → **ViT family (ViT, DeiT, Swin v1/v2,
  MAE, BEiT, DINOv2/DINOv3)** → object detection (Faster R-CNN, YOLO v1–v12,
  DETR, Grounding DINO) → segmentation (U-Net, DeepLab, **SAM/SAM 2**) →
  3D vision (NeRF, **3DGS/4DGS**) → video (SlowFast, TimeSformer, VideoMAE,
  Sora/Veo/Kling).
- **Speech & Audio (37 topics)** — STFT/mel/MFCC → CTC, RNN-T, Conformer →
  **Wav2Vec 2.0, HuBERT, Whisper v3** → TTS (Tacotron → VITS → VALL-E,
  Voicebox, XTTS) → neural codecs (SoundStream, Encodec).
- **Multimodal (62 topics)** — encoders + projectors + fusion strategies →
  **CLIP / SigLIP / SigLIP2** → **BLIP-2, LLaVA family, Flamingo, Qwen2.5-VL,
  InternVL, Pixtral, Molmo, Janus** → audio-LMs (Qwen2-Audio, SALMONN) →
  video-LMs → **omni / any-to-any (GPT-4o, Gemini native, NExT-GPT, CoDi-2,
  Unified-IO 2)** → **VLA / embodied (RT-2, OpenVLA, π0, V-JEPA 2)**.
- **Multimodal training pipeline** — alignment stage → multimodal pretraining
  → multimodal SFT → multimodal DPO/GRPO.

**Exit criterion:** train a CLIP-style dual-encoder from scratch on a small
image-text dataset; then build a LLaVA-style projector that bolts a vision
encoder onto a small pretrained LLM and instruction-tune it.

### Phase 7 — ML Systems, MLOps, Eval, Safety (6–10 weeks)

**Goal:** ship and operate.

- **Distributed training** — DP/DDP → tensor (Megatron-style) → pipeline
  (GPipe/1F1B/interleaved) → sequence → **context (ring attention)** →
  expert parallelism → **ZeRO 1/2/3, FSDP/FSDP2, DeepSpeed**, 3D/4D/5D
  parallelism, NCCL collectives.
- **Hardware** — A100/H100/H200/B200/GB200, TPUs, Trainium, memory
  hierarchies, NVLink/NVSwitch, InfiniBand.
- **Mixed precision** — FP16 + loss scaling, BF16, FP8, microscaling.
- **Memory optimization** — activation checkpointing, selective recomputation,
  CPU/NVMe offload, paged optimizers.
- **Data engineering** — Common Crawl/FineWeb/RedPajama/Dolma pipelines,
  dedup (MinHash/SimHash), quality filtering, decontamination, packing,
  WebDataset/Mosaic Streaming.
- **Serving** — vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, Triton Inference
  Server, multi-LoRA serving, disaggregated prefill/decode, edge deployment.
- **MLOps** — feature stores, registries, CI/CD for ML, drift detection,
  LLM observability (LangSmith, Langfuse, Helicone), A/B testing.
- **Evaluation (59 topics)** — MMLU, GPQA, AIME, HLE, ARC-AGI for reasoning;
  HumanEval/SWE-bench/LiveCodeBench for code; BFCL/τ-Bench/GAIA for agents;
  MMMU/MMBench/Video-MME for multimodal; needle/RULER/LongBench for long-context;
  LMArena Elo; LLM-as-a-judge methodology, calibration, contamination.
- **AI Safety & Alignment (46 topics)** — alignment theory (outer/inner,
  mesa-opt, scalable oversight, weak-to-strong), Constitutional AI, red
  teaming, jailbreaks (GCG, PAIR, many-shot, prompt injection), refusal
  training, bias/fairness, differential privacy (DP-SGD), adversarial
  robustness, watermarking, governance (EU AI Act, NIST RMF, RSPs).

**Exit criterion:** run a multi-GPU FSDP fine-tune; deploy to vLLM with
INT4 quant; build an evaluation harness with at least 3 benchmarks +
LLM-as-judge; run a basic red-team and document findings.

### Phase 8 — Frontier, Research, Capstone (ongoing, 6+ months)

**Goal:** original work.

- **Emerging topics (80 topics)** — SSMs/Mamba-2, hybrids, diffusion LMs,
  byte-latent transformers, V-JEPA 2 world models, neural ODEs, GNNs,
  geometric DL, neuro-symbolic, AI for science (AlphaFold 3, ESM 3,
  GraphCast, AlphaProof), causal ML.
- **Research skills (36 topics)** — arXiv triage, paper reproduction,
  LaTeX, conference fluency (NeurIPS, ICML, ICLR, ACL, CVPR, MLSys),
  ablation design, rebuttal writing.
- **Advanced math you deferred** — measure theory, functional analysis,
  Itô calculus (for diffusion theory), differential geometry (for geometric
  DL and equivariance).
- **Capstone (46 topics, 10 stages)** — execute the compass's Stage 0–10
  pipeline: plan → data → tokenizer/encoders → architecture → pretraining →
  multimodal alignment → multimodal pretraining → SFT + preference tuning →
  evaluation → deployment → iteration.

**Exit criterion:** publish a workshop paper, reproduce a 2024–2026 SOTA
paper end-to-end, or ship a multimodal model that someone actually uses.

---

## Part B — 30-Day Sprint (Operational Fluency)

**Premise:** 30 days × ~7 hours/day = ~210 hours. This is **not** the PhD
roadmap above compressed — it is a different document. It targets the
shortest path from zero to "can build, train, fine-tune, evaluate, and
serve a small modern transformer end-to-end, and understand every line."

**Daily structure (~7 hrs):**
- **2 hrs Theory** — generate the day's chapter using
  `deep_research_system_prompt.md` (paste the topic name as `<TOPIC>`),
  read it actively, take notes.
- **3 hrs Code** — implement the day's deliverable from scratch in PyTorch.
- **1 hr Paper / video** — one canonical paper or one lecture.
- **1 hr Review** — re-derive the morning's math by hand; update a running
  glossary; commit notes to a personal repo.

**Stack to install on Day 0:** Python 3.11, PyTorch (CUDA build), NumPy,
einops, Hugging Face (transformers/datasets/tokenizers/accelerate/peft/trl),
W&B, Jupyter, pytest, ruff. One GPU (rent an H100/A100 hour by hour from
Lambda/RunPod/Modal as needed; days 18–28 will need GPU time).

---

### Week 1 — Math + Tools + Classical ML you can't skip (Days 1–7)

The aim is *just enough* math and ML to make the deep-learning weeks land.
You will not learn measure theory. You will learn what a gradient is.

| Day | Topic (feed to research prompt) | Code deliverable |
|---|---|---|
| **1** | *Scalars, vectors, matrices, tensors* + *Matrix operations* + *Eigenvalues/SVD* | NumPy: implement matmul, transpose, pseudoinverse, power-iteration eigenvector. |
| **2** | *Multivariable calculus* + *Chain rule* + *Backpropagation as reverse-mode autodiff* | Build a 30-line scalar autograd engine (Karpathy's `micrograd`). Train a 2-layer MLP on XOR with it. |
| **3** | *Probability* (random vars, expectation, Gaussian, MLE/MAP) + *Cross-entropy & KL divergence* | NumPy: implement logistic regression with MLE; derive the loss gradient on paper, code it, verify against autograd. |
| **4** | *Optimization*: SGD, momentum, Adam, AdamW + *Loss landscape* | Implement SGD, Momentum, Adam from scratch as PyTorch optimizers; reproduce on a toy convex + non-convex problem. |
| **5** | *PyTorch fundamentals*: `Tensor`, `autograd`, `nn.Module`, `DataLoader`, `optim`, mixed precision | Reimplement Day 2's MLP in idiomatic PyTorch; add a `DataLoader`, mixed-precision `autocast`, W&B logging, checkpointing. |
| **6** | *Bias-variance*, *cross-validation*, *regularization* (L1/L2, dropout, early stopping) + *Decision trees → Gradient boosting (XGBoost)* | Tabular task (sklearn dataset): baseline → GBDT → tune. Plot bias-variance vs depth. |
| **7** | *CNNs from scratch*: convolution, padding, stride, pooling, receptive field; *LeNet → ResNet* | Implement Conv2d + BatchNorm from primitives (no `nn.Conv2d`), build ResNet-18 by hand, train on CIFAR-10 to >85%. |

**End-of-week checkpoint:** can you (a) derive backprop for an MLP, (b) write
training loops in PyTorch fluently, (c) explain why Adam is the default?
If "no," spend a buffer day before continuing.

---

### Week 2 — Deep Learning + Sequence Models + Attention (Days 8–14)

Earn the transformer. Don't skip RNNs — without them attention feels like magic.

| Day | Topic | Deliverable |
|---|---|---|
| **8** | *Normalization* (BatchNorm vs LayerNorm vs RMSNorm) + *Weight initialization* (Xavier, He) + *Activation functions* (ReLU → GELU → SwiGLU) | Implement all three norms from scratch; verify gradients match PyTorch's. |
| **9** | *Autoencoders* + *VAE* + *VQ-VAE* (latent representations, ELBO derivation) | Train a VAE on MNIST; visualize latent space; train VQ-VAE; explain reparameterization trick. |
| **10** | *Diffusion Models*: DDPM + DDIM + classifier-free guidance + *Flow matching* (current frontier) | Implement DDPM from scratch on MNIST (forward/reverse process, noise schedule, U-Net). Sample images. |
| **11** | *RNN, LSTM, GRU* + *BPTT* + *vanishing/exploding gradients* | Implement an LSTM cell from scratch; train a char-RNN on a text file. |
| **12** | *Encoder-decoder seq2seq* + *Bahdanau/Luong attention* + *Beam search* | Build a tiny seq2seq+attention translator (toy dataset). Implement beam-search decoding. |
| **13** | *Scaled dot-product attention* + *Multi-head attention* + *MQA, GQA, MLA* | Implement MHA from scratch (no `nn.MultiheadAttention`); verify against PyTorch. Add GQA variant. |
| **14** | *Tokenization*: BPE, byte-BPE, SentencePiece, Unigram | Train a BPE tokenizer on a small corpus from scratch (no library). Then compare against HF `tokenizers`. |

**End-of-week checkpoint:** you have hand-rolled attention, LSTM, VAE, DDPM,
and BPE. You are ready for transformers.

---

### Week 3 — Transformers, Pretraining, Fine-Tuning, Alignment (Days 15–21)

This is the hottest week. Slow down if you need to.

| Day | Topic | Deliverable |
|---|---|---|
| **15** | *Transformer architecture* (Vaswani) — full block: pre-norm, residual, MLP, SwiGLU, RMSNorm + *Positional encoding*: sinusoidal → **RoPE** → ALiBi | Build a nanoGPT-style decoder transformer from scratch (~250 lines). Implement RoPE. Train on TinyShakespeare to overfitting. |
| **16** | *Pretraining objectives*: CLM, MLM, span corruption, FIM + *Scaling laws*: Kaplan → Chinchilla → compute-optimal | Pretrain your Day-15 model on TinyStories (~10M tokens). Plot loss vs compute. Pick a Chinchilla-optimal size. |
| **17** | *LLM architectures* survey: BERT, T5, GPT-2/3, **LLaMA 3, Mistral, Qwen, DeepSeek-V3, Gemma** + *MoE* (Mixtral, DeepSeek-MoE) | Read LLaMA 3 + DeepSeek-V3 papers. Modify your nanoGPT to LLaMA-style (RMSNorm + SwiGLU + RoPE + GQA). Train. |
| **18** | *SSMs (Mamba/Mamba-2)* + *Linear attention (RWKV, RetNet)* + *Hybrids (Jamba, Granite 4)* | Read Mamba + Mamba-2 papers. Implement a single S6/Mamba block (or use `mamba-ssm`). Compare perplexity vs your transformer. |
| **19** | *Instruction tuning / SFT* + *PEFT*: **LoRA, QLoRA, DoRA**, prefix/prompt tuning, IA³ | LoRA-fine-tune a small open model (TinyLlama or Qwen-0.5B) on Alpaca with HF `peft`. Compare to full fine-tune. |
| **20** | *RLHF pipeline* (SFT → reward model → PPO) + *DPO* + *GRPO, DAPO, RLVR* (2024–2026 frontier) + Constitutional AI | DPO-tune yesterday's model on a preference dataset (TRL `DPOTrainer`). Read DPO + GRPO papers. Explain why DPO replaced PPO for most teams. |
| **21** | *Reasoning at inference*: CoT, self-consistency, ToT, ReAct, process reward models, **long-CoT (o1, R1, QwQ)** + *Decoding*: greedy, beam, top-k/p, min-p, contrastive | Implement temperature/top-k/top-p/min-p sampling from logits by hand. Reproduce a CoT prompt benchmark on GSM8K. |

**End-of-week checkpoint:** you have pretrained, SFT'd, and DPO'd a small
LLM you built from scratch.

---

### Week 4 — Multimodal, Systems, Eval, Capstone (Days 22–30)

| Day | Topic | Deliverable |
|---|---|---|
| **22** | *RAG*: dense + sparse + hybrid + rerankers + vector DBs (FAISS, Qdrant) + ANN (HNSW) | Build a RAG pipeline: chunk docs → embed (BGE or E5) → FAISS index → rerank (cross-encoder) → generate. |
| **23** | *Agents & tool use*: function calling, **MCP**, ReAct, multi-agent | Build a small ReAct agent with tool calling (calculator + web). Wire one MCP server. |
| **24** | *Long context*: sliding window, **FlashAttention 1/2/3**, ring attention, **PagedAttention**, KV cache + StreamingLLM | Read FlashAttention-2 paper. Profile attention memory in your Day-15 model with and without `F.scaled_dot_product_attention`. |
| **25** | *Inference optimization*: quantization (INT8/INT4, GPTQ, AWQ, **FP8**, MXFP4) + **speculative decoding** (EAGLE, Medusa) + engines (vLLM, SGLang, TensorRT-LLM) | Quantize your DPO'd model to INT4 (bitsandbytes or AutoGPTQ). Serve with vLLM. Measure tokens/sec before and after. |
| **26** | *Distributed training*: DDP → FSDP/ZeRO 1/2/3 → tensor/pipeline parallelism + mixed precision (BF16, FP8) | Run an FSDP fine-tune on 2 GPUs (rent if needed). Profile memory. Read the ZeRO paper. |
| **27** | *Vision*: **ViT, DINOv2/DINOv3, SAM 2** + *Multimodal*: **CLIP, SigLIP2, LLaVA, Qwen2.5-VL, GPT-4o-style omni** + multimodal training pipeline | Train a tiny CLIP from scratch on a small image-text dataset (1k pairs). Compute zero-shot accuracy on a held-out task. |
| **28** | *LLaVA-style multimodal*: vision encoder + projector + frozen LLM, then unfreeze | Bolt a SigLIP encoder + 2-layer MLP projector onto your Day-19 LoRA-tuned LLM. Train the projector on ~1k caption pairs. |
| **29** | *Evaluation*: MMLU, GSM8K, HumanEval, **MMMU**, **LMArena Elo**, **LLM-as-judge**, calibration, contamination + *Safety*: **jailbreaks (GCG, PAIR, many-shot, prompt injection)**, refusal training, **Constitutional AI**, red teaming, watermarking | Evaluate your model on GSM8K + a small custom eval. Set up LLM-as-judge with pairwise comparison. Try 3 jailbreaks and log refusal rate. |
| **30** | **Capstone integration day** (no new topics) | Wire everything: pretrained model → SFT → DPO → RAG + tool-calling agent → INT4 quant → vLLM serve → eval harness → red-team report. Write a 2-page architecture doc. |

---

## How to Use the Repo's Research Prompts

The system prompt (`deep_research_system_prompt.md`) is your textbook generator.
Workflow for each day:

1. Open the prompt file. Replace every `<TOPIC>` placeholder with the day's
   topic name (verbatim from `ml_llm_topics.csv` is fine).
2. Send the whole prompt as a single message to a capable model (Claude
   Opus / Sonnet, GPT-5, Gemini 2.x with web search enabled).
3. You'll get a 30–80 page chapter following the 15-section structure. Read
   actively: do the worked examples by hand, answer the self-assessment
   questions without peeking, then check.
4. Save the chapter to `notes/<phase>/<topic>.md` in your personal repo.
   By day 30 you'll have ~30 chapters — your personalized textbook.

Use the **compact prompt** for shorter topics (a single algorithm, a single
distribution); use the **full prompt** for chapter-scale topics
("Transformer architecture", "Diffusion models").

---

## What This Plan Deliberately Leaves Out

To set expectations:

- **Measure theory, functional analysis, Itô calculus, differential geometry.**
  Needed only for theoretical research in diffusion, generalization, and
  geometric DL. Phase 8.
- **JAX/Flax, CUDA C++, Triton kernels, ROCm.** PyTorch is enough for 30
  days. Triton/CUDA arrive when you're optimizing real kernels.
- **Classical CV depth** (SIFT, HOG, multi-view geometry, NeRF, 3DGS).
  Skimmed in Week 4. A vision specialization would add 4–6 weeks.
- **Classical RL** (DQN, A3C, SAC, MuZero, model-based RL). RLHF/DPO are
  what you need for LLMs; full RL specialization is Berkeley CS285 territory.
- **Speech & audio depth.** Whisper as a black box is enough; full
  TTS/ASR/codec specialization adds 3–4 weeks.
- **Quantum ML, neuro-symbolic, GNNs, causal inference, AI-for-science.**
  All in the compass; none on the critical path to a multimodal LLM.

After Day 30, the right move is to pick **one** Phase 6–8 specialization
(vision, audio, RL, systems, interpretability, alignment) and go deep for
3–6 months while continuing to ship.
