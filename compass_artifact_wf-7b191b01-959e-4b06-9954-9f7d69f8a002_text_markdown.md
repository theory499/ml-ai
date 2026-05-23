# Comprehensive Topic Checklist: From Zero ML Knowledge to PhD-Level Multimodal LLM Architect

## TL;DR
- This is an exhaustive, hierarchical checklist of every topic a backend developer with limited math background must master to reach PhD-level depth as an LLM architect capable of building a multimodal LLM from scratch.
- Topics are grouped into 16 top-level domains spanning mathematical foundations, programming/systems, classical ML, deep learning, transformers/LLMs, NLP, computer vision, speech/audio, multimodal learning, ML systems & MLOps, research skills, evaluation, safety/alignment, and emerging/advanced research areas.
- Coverage includes both classical foundations (linear algebra, calculus, CNNs, RNNs, RL) and current 2025–2026 frontier topics (Mamba/SSMs, MoE, GRPO/DAPO/RLVR, FlashAttention-3, FP8/MXFP4 quantization, V-JEPA 2 world models, MCP agents, sparse autoencoders, rectified flow matching, DINOv3, etc.).

## Key Findings
- A complete curriculum spans roughly 16 top-level domains, each with multiple sub-domains and an explicit mathematical-prerequisites layer.
- Modern (2025–2026) LLM architecture work requires fluency in transformer internals, hybrid SSM/attention architectures, MoE, distributed training (3D/4D parallelism, ZeRO/FSDP), and inference systems (vLLM, paged attention, speculative decoding).
- Multimodal mastery requires separate competence in vision, audio, video, and 3D representations, plus cross-modal fusion, contrastive alignment, and unified any-to-any architectures.
- PhD-level depth additionally demands research methodology, paper reproduction, conference fluency (NeurIPS/ICML/ICLR/ACL/CVPR), evaluation rigor, and AI safety/alignment literacy.

## Details

## 1. Mathematical Foundations

- **Linear Algebra**
  - Scalars, vectors, matrices, tensors
  - Vector spaces, subspaces, basis, dimension, span
  - Linear independence and rank
  - Inner products, norms (L1, L2, Lp, Frobenius, nuclear, spectral)
  - Orthogonality, orthonormal bases, Gram-Schmidt
  - Matrix operations (addition, multiplication, transpose, inverse, pseudoinverse)
  - Determinants and trace
  - Eigenvalues and eigenvectors
  - Diagonalization
  - Symmetric, Hermitian, orthogonal, unitary, positive-definite matrices
  - Matrix decompositions (LU, QR, Cholesky, Schur, Jordan)
  - Singular Value Decomposition (SVD) and truncated SVD
  - Spectral theorem
  - Tensor algebra and operations (contractions, outer products, Kronecker, Hadamard)
  - Einstein summation notation (einsum)
  - Tensor decompositions (CP, Tucker, tensor train)
  - Sparse matrices and sparse linear algebra
  - Block matrices and partitioned operations
  - Matrix calculus identities
  - Numerical linear algebra (iterative solvers, Krylov methods, Lanczos, Arnoldi)

- **Calculus**
  - Limits, continuity, sequences, series
  - Differential calculus (derivatives, rules, higher-order derivatives)
  - Integral calculus (definite, indefinite, techniques of integration)
  - Multivariable calculus
  - Partial derivatives
  - Gradients, directional derivatives
  - Jacobian matrices
  - Hessian matrices and second-order derivatives
  - Chain rule (scalar and multivariable)
  - Vector calculus (divergence, curl, line/surface integrals)
  - Taylor series and Taylor expansions
  - Implicit function theorem
  - Inverse function theorem
  - Lagrange multipliers
  - Variational calculus
  - Backpropagation as reverse-mode automatic differentiation
  - Forward-mode and reverse-mode autodiff
  - Differentiation through optimization (implicit differentiation)

- **Probability Theory**
  - Sample spaces, events, σ-algebras
  - Axioms of probability
  - Conditional probability and independence
  - Bayes' theorem
  - Random variables (discrete, continuous)
  - Probability mass functions and density functions
  - Cumulative distribution functions
  - Joint, marginal, and conditional distributions
  - Expectation, variance, covariance, correlation
  - Moments and moment-generating functions
  - Characteristic functions
  - Common distributions (Bernoulli, binomial, Poisson, Gaussian, multivariate Gaussian, exponential, Beta, Dirichlet, Gamma, Categorical, Multinomial, Laplace, Cauchy, Student-t)
  - Mixture models
  - Exponential family
  - Conjugate priors
  - Maximum Likelihood Estimation (MLE)
  - Maximum A Posteriori (MAP) estimation
  - Bayesian inference
  - Law of large numbers
  - Central limit theorem
  - Concentration inequalities (Markov, Chebyshev, Hoeffding, Chernoff)
  - Importance sampling, rejection sampling
  - Change of variables in probability

- **Statistics**
  - Descriptive statistics
  - Sampling theory and sampling distributions
  - Point estimation and interval estimation
  - Confidence intervals
  - Hypothesis testing (t-test, z-test, χ², ANOVA, F-test)
  - p-values, type I/II errors, statistical power
  - Multiple testing correction (Bonferroni, FDR)
  - Nonparametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis)
  - Bootstrap and resampling
  - Bayesian statistics
  - Linear regression theory
  - Generalized linear models
  - Bias-variance decomposition
  - Cross-validation theory
  - Order statistics
  - Robust statistics

- **Information Theory**
  - Self-information and Shannon entropy
  - Joint, conditional, and differential entropy
  - Cross-entropy
  - Kullback-Leibler (KL) divergence
  - Jensen-Shannon divergence
  - Mutual information and pointwise mutual information
  - f-divergences and Wasserstein distance
  - Source coding theorem
  - Channel capacity and noisy channel coding
  - Rate-distortion theory
  - Minimum description length
  - Variational bounds (ELBO)

- **Optimization Theory**
  - Convex sets and convex functions
  - Convex optimization
  - Non-convex optimization landscapes
  - Lagrangian duality
  - Karush-Kuhn-Tucker (KKT) conditions
  - Constrained vs unconstrained optimization
  - Linear, quadratic, and semidefinite programming
  - Gradient descent and variants
  - Stochastic Gradient Descent (SGD) and minibatch SGD
  - Momentum, Nesterov momentum
  - Adaptive methods (Adagrad, RMSprop, Adam, AdamW, AdaBelief, Lion, Sophia, Shampoo)
  - Second-order methods (Newton, quasi-Newton, BFGS, L-BFGS, K-FAC)
  - Proximal methods, ADMM
  - Coordinate descent
  - Trust region methods
  - Line search methods
  - Saddle points and escape strategies
  - Loss landscape geometry
  - Conditioning and preconditioning
  - Convergence analysis and rates
  - Mirror descent and natural gradient
  - Bayesian optimization
  - Evolutionary strategies and CMA-ES

- **Numerical Methods and Numerical Analysis**
  - Floating-point representation (FP64, FP32, FP16, BF16, FP8, FP4)
  - Numerical stability and conditioning
  - Condition numbers
  - Catastrophic cancellation
  - Forward/backward error analysis
  - Numerical integration (quadrature)
  - Numerical solution of linear systems
  - Iterative methods (Jacobi, Gauss-Seidel, conjugate gradient)
  - Numerical differentiation
  - Root finding
  - Stochastic numerical methods

- **Discrete Mathematics**
  - Set theory
  - Logic and propositional/predicate calculus
  - Combinatorics and counting
  - Permutations and combinations
  - Graph theory (directed, undirected, weighted, trees, DAGs)
  - Graph algorithms (BFS, DFS, shortest paths, MST, flows)
  - Spectral graph theory
  - Boolean algebra
  - Recurrence relations
  - Generating functions
  - Number theory basics

- **Differential Equations**
  - Ordinary differential equations (ODEs)
  - Numerical ODE solvers (Euler, Runge-Kutta, adaptive solvers)
  - Partial differential equations (PDEs)
  - Stochastic differential equations (SDEs)
  - Neural ODEs and continuous-depth models
  - Diffusion equation and Fokker-Planck

- **Functional Analysis (Foundational)**
  - Metric spaces, Banach spaces, Hilbert spaces
  - Function spaces (L², Sobolev)
  - Operator theory
  - Reproducing Kernel Hilbert Spaces (RKHS)
  - Spectral theory of operators
  - Distributions and generalized functions

- **Measure Theory (Foundational)**
  - σ-algebras and measurable spaces
  - Lebesgue measure and integration
  - Radon-Nikodym derivative
  - Probability as measure theory
  - Convergence theorems (monotone, dominated)

- **Stochastic Processes**
  - Markov chains (discrete and continuous)
  - Hidden Markov Models
  - Markov Chain Monte Carlo (MCMC)
  - Metropolis-Hastings, Gibbs sampling, HMC, NUTS
  - Brownian motion and Wiener processes
  - Itô calculus
  - Poisson processes
  - Martingales
  - Stochastic differential equations for diffusion models
  - Score-based generative modeling foundations

- **Game Theory**
  - Normal-form games and extensive-form games
  - Nash equilibrium
  - Minimax theorem
  - Zero-sum and general-sum games
  - Cooperative game theory
  - Mechanism design basics
  - Multi-agent decision theory
  - Game-theoretic foundations of GANs
  - Multi-agent reinforcement learning foundations

- **Topology and Differential Geometry (Advanced)**
  - Manifolds and tangent spaces
  - Riemannian geometry
  - Lie groups and Lie algebras
  - Group theory (symmetry, equivariance)
  - Information geometry

---

## 2. Programming and Tools Foundations

- **Python Ecosystem**
  - Python language (advanced features, generators, decorators, context managers, async)
  - Type hints and typing module
  - NumPy
  - Pandas
  - Matplotlib, Seaborn, Plotly
  - SciPy
  - Scikit-learn
  - Jupyter, IPython, JupyterLab
  - Hugging Face ecosystem (transformers, datasets, tokenizers, accelerate, PEFT, TRL, evaluate)
  - einops

- **Deep Learning Frameworks**
  - PyTorch (autograd, nn, optim, distributed)
  - PyTorch Lightning
  - TensorFlow and Keras
  - JAX, Flax, Haiku, Equinox
  - ONNX
  - TorchScript and torch.compile
  - Triton (OpenAI Python kernel DSL)

- **GPU and Hardware Programming**
  - CUDA programming model
  - cuBLAS, cuDNN, CUTLASS
  - Triton kernels
  - NCCL communication library
  - Memory hierarchy (registers, shared, L1/L2, HBM)
  - Tensor cores and warp-level primitives
  - Mixed precision and tensor core programming
  - Profilers (Nsight, NVProf, PyTorch Profiler)
  - ROCm/HIP for AMD GPUs
  - TPU programming via JAX/XLA

- **Distributed Computing Frameworks**
  - Ray, Ray Train, Ray Serve, Ray Tune
  - Dask
  - Apache Spark / PySpark
  - Hadoop / HDFS
  - MPI

- **Version Control and Experiment Tracking**
  - Git, GitHub/GitLab workflows
  - Git LFS
  - DVC (Data Version Control)
  - MLflow
  - Weights & Biases
  - TensorBoard
  - Neptune.ai, Comet
  - Hydra and OmegaConf for configuration

- **Data Engineering Tools**
  - Apache Spark
  - Apache Beam
  - Apache Arrow / Parquet
  - Apache Kafka
  - Airflow / Prefect / Dagster
  - SQL and NoSQL databases
  - Object stores (S3, GCS)

- **Containerization and Deployment**
  - Docker
  - Kubernetes
  - Helm
  - Serverless GPUs
  - CI/CD (GitHub Actions, Jenkins)

- **C++ for ML Systems**
  - Modern C++ (C++17/20)
  - Templates and metaprogramming
  - Memory management
  - PyTorch C++ frontend (LibTorch)
  - Custom CUDA C++ extensions
  - pybind11

- **Software Engineering for ML**
  - Testing (pytest, hypothesis)
  - Profiling and benchmarking
  - Logging and observability
  - Code style, linting, formatting
  - Reproducibility practices

---

## 3. Classical Machine Learning

- **ML Foundations**
  - Learning paradigms (supervised, unsupervised, semi-supervised, self-supervised, reinforcement)
  - Hypothesis space, inductive bias
  - Empirical risk minimization
  - PAC learning and VC dimension
  - Bias-variance tradeoff
  - Overfitting, underfitting, generalization
  - Cross-validation strategies
  - Feature engineering and selection
  - Curse of dimensionality
  - No free lunch theorem

- **Supervised Learning**
  - Linear regression
  - Polynomial regression
  - Ridge regression
  - Lasso regression
  - Elastic Net
  - Logistic regression (binary and multinomial)
  - Probit regression
  - Generalized linear models
  - Linear and Quadratic Discriminant Analysis
  - k-Nearest Neighbors
  - Naive Bayes (Gaussian, Multinomial, Bernoulli)
  - Decision trees (ID3, C4.5, CART)
  - Random forests
  - Extra Trees
  - Gradient boosting machines
  - XGBoost
  - LightGBM
  - CatBoost
  - Support Vector Machines (linear, kernel, soft-margin)
  - Kernel methods and the kernel trick
  - Gaussian Processes
  - Bayesian linear/logistic regression
  - Perceptron algorithm

- **Ensemble Methods**
  - Bagging
  - Boosting (AdaBoost, gradient boosting)
  - Stacking
  - Voting (hard/soft)
  - Blending
  - Snapshot ensembles

- **Unsupervised Learning**
  - Clustering
    - K-means, K-medoids, K-means++
    - Hierarchical clustering (agglomerative, divisive)
    - DBSCAN, OPTICS, HDBSCAN
    - Gaussian Mixture Models (EM algorithm)
    - Spectral clustering
    - Mean shift
    - Affinity propagation
  - Dimensionality Reduction
    - PCA, kernel PCA, probabilistic PCA
    - ICA
    - LDA (Linear Discriminant Analysis)
    - t-SNE
    - UMAP
    - MDS, Isomap, LLE
    - Random projections
    - Autoencoders for dimensionality reduction
    - Matrix factorization (NMF, SVD-based)
  - Anomaly Detection
    - Isolation Forest
    - One-class SVM
    - Local Outlier Factor
    - Autoencoder-based anomaly detection
  - Association Rule Learning (Apriori, FP-growth)
  - Density Estimation (KDE, parametric)
  - Topic Modeling (LDA, NMF)

- **Probabilistic Graphical Models**
  - Bayesian networks
  - Markov random fields
  - Conditional random fields
  - Inference (variable elimination, belief propagation, junction tree)
  - Approximate inference (MCMC, variational)
  - Structure learning

- **Semi-Supervised Learning**
  - Self-training and pseudo-labeling
  - Co-training
  - Graph-based semi-supervised methods
  - Consistency regularization (FixMatch, MixMatch)

- **Self-Supervised Learning Foundations**
  - Pretext tasks
  - Contrastive learning principles
  - Predictive learning principles

- **Reinforcement Learning**
  - Markov Decision Processes
  - Bellman equations
  - Dynamic programming (policy iteration, value iteration)
  - Monte Carlo methods
  - Temporal difference learning
  - Q-learning, SARSA, Expected SARSA
  - Function approximation in RL
  - Deep Q-Networks (DQN, Double DQN, Dueling DQN, Rainbow)
  - Policy gradient theorem
  - REINFORCE
  - Actor-critic methods
  - A2C, A3C
  - TRPO, PPO
  - DDPG, TD3
  - Soft Actor-Critic (SAC)
  - Distributional RL (C51, QR-DQN)
  - Model-based RL (Dyna, MuZero, Dreamer)
  - World models for RL
  - Exploration strategies (ε-greedy, UCB, Thompson sampling, intrinsic motivation, RND)
  - Multi-agent RL (MADDPG, QMIX, self-play)
  - Inverse RL
  - Imitation learning and behavior cloning
  - DAgger
  - Hierarchical RL
  - Offline / batch RL (CQL, IQL, AWR)
  - Goal-conditioned RL
  - Reward shaping
  - RLHF (Reinforcement Learning from Human Feedback)
  - RLAIF (Reinforcement Learning from AI Feedback)
  - RLVR (Reinforcement Learning with Verifiable Rewards)
  - GRPO, DAPO, REINFORCE++

- **Other Learning Paradigms**
  - Online learning and regret minimization
  - Multi-armed bandits, contextual bandits
  - Active learning
  - Transfer learning
  - Domain adaptation
  - Meta-learning (MAML, Reptile, ProtoNets)
  - Few-shot, one-shot, zero-shot learning
  - Curriculum learning
  - Continual / lifelong learning
  - Federated learning

---

## 4. Deep Learning Fundamentals

- **Artificial Neural Networks**
  - Biological inspiration and history
  - Perceptron and multilayer perceptron (MLP)
  - Universal approximation theorem
  - Activation functions (sigmoid, tanh, ReLU, Leaky ReLU, PReLU, ELU, SELU, GELU, Swish/SiLU, Mish, Softplus, Softmax)
  - Output layers for classification/regression
  - Loss functions
    - MSE, MAE, Huber
    - Cross-entropy (binary, categorical, sparse categorical)
    - Hinge loss
    - Focal loss
    - Contrastive, triplet, InfoNCE
    - CTC loss
    - Wasserstein loss
    - Custom and composite losses
  - Forward and backward propagation
  - Computational graphs
  - Automatic differentiation (forward and reverse mode)
  - Weight initialization (Xavier/Glorot, He/Kaiming, orthogonal, LSUV)
  - Regularization
    - L1, L2 (weight decay)
    - Dropout, DropConnect, DropPath, Stochastic Depth
    - Early stopping
    - Data augmentation
    - Label smoothing
    - Mixup, CutMix, Manifold Mixup
  - Normalization
    - Batch Normalization
    - Layer Normalization
    - Instance Normalization
    - Group Normalization
    - RMSNorm
    - Weight Normalization
    - Spectral Normalization
  - Optimization Algorithms
    - SGD with momentum / Nesterov
    - Adagrad, Adadelta, RMSprop
    - Adam, AdamW, NAdam, AMSGrad
    - Lion, Sophia, Shampoo
    - LARS, LAMB
    - Lookahead, RAdam
  - Learning Rate Scheduling
    - Step decay, exponential decay
    - Cosine annealing, cosine with warm restarts
    - Linear warmup
    - 1cycle policy
    - Cyclical learning rates
    - ReduceLROnPlateau
  - Hyperparameter Tuning
    - Grid search, random search
    - Bayesian optimization
    - Hyperband, BOHB
    - Population-based training
    - Optuna, Ray Tune
  - Gradient Issues
    - Vanishing gradients
    - Exploding gradients
    - Gradient clipping
    - Gradient accumulation
  - Skip / residual connections

- **Convolutional Neural Networks**
  - Convolution operation (1D, 2D, 3D)
  - Cross-correlation vs convolution
  - Padding (valid, same, causal)
  - Strides
  - Dilated/atrous convolution
  - Transposed/deconvolution
  - Depthwise and depthwise-separable convolution
  - Grouped convolution
  - 1x1 convolutions
  - Pooling (max, average, global, adaptive)
  - Receptive fields
  - Feature maps and channels
  - Classic Architectures
    - LeNet-5
    - AlexNet
    - VGG (16, 19)
    - GoogLeNet / Inception (v1–v4, Inception-ResNet)
    - ResNet, ResNeXt, Wide ResNet
    - DenseNet
    - SqueezeNet
    - MobileNet (v1, v2, v3)
    - ShuffleNet
    - EfficientNet (v1, v2)
    - RegNet
    - ConvNeXt, ConvNeXt v2

- **Recurrent Neural Networks**
  - Vanilla RNN
  - Backpropagation Through Time (BPTT)
  - Truncated BPTT
  - Vanishing/exploding gradient in RNNs
  - LSTM (cells, gates, peephole)
  - GRU
  - Bidirectional RNNs
  - Deep / stacked RNNs
  - Encoder-decoder / sequence-to-sequence
  - Teacher forcing and exposure bias
  - Beam search decoding

- **Attention Mechanisms (Pre-Transformer)**
  - Bahdanau (additive) attention
  - Luong (multiplicative) attention
  - Self-attention
  - Multi-head attention
  - Hard vs soft attention

- **Autoencoders**
  - Vanilla autoencoders
  - Undercomplete and overcomplete autoencoders
  - Denoising autoencoders
  - Sparse autoencoders
  - Contractive autoencoders
  - Variational Autoencoders (VAE, β-VAE, VQ-VAE, VQ-VAE-2)
  - Discrete latent variable models

- **Generative Models**
  - Generative Adversarial Networks (GANs)
    - Vanilla GAN
    - DCGAN
    - Conditional GAN (cGAN)
    - WGAN, WGAN-GP
    - LSGAN, BEGAN
    - Progressive GAN
    - StyleGAN, StyleGAN2, StyleGAN3
    - BigGAN
    - CycleGAN, Pix2Pix
    - SRGAN, ESRGAN
    - Mode collapse and training stability
  - Variational Autoencoders (advanced)
  - Normalizing Flows (RealNVP, Glow, NICE, MAF, IAF, neural spline flows)
  - Autoregressive models (PixelRNN, PixelCNN, MADE)
  - Energy-Based Models
  - Diffusion Models
    - DDPM
    - DDIM
    - Score-based models (NCSN)
    - SDE-based diffusion
    - Latent diffusion
    - Classifier-free guidance
    - Consistency models
  - Flow Matching
    - Conditional flow matching
    - Rectified flow
    - Reflow / rectified diffusion

- **Theoretical Concepts**
  - Lottery ticket hypothesis
  - Double descent
  - Implicit regularization
  - Neural Tangent Kernel
  - Information bottleneck

---

## 5. Transformers and Large Language Models

- **Transformer Architecture Fundamentals**
  - Scaled dot-product attention
  - Self-attention
  - Multi-head attention
  - Multi-Query Attention (MQA)
  - Grouped-Query Attention (GQA)
  - Multi-Latent Attention (MLA)
  - Cross-attention
  - Causal / masked attention
  - Positional Encodings
    - Sinusoidal positional encoding
    - Learned positional embeddings
    - Relative positional encoding
    - Rotary Position Embeddings (RoPE)
    - ALiBi
    - YaRN, NTK-aware scaling
  - Feedforward / MLP blocks
  - SwiGLU and gated activations
  - Residual connections
  - Pre-norm vs post-norm
  - Layer normalization and RMSNorm
  - Encoder-only architectures
  - Decoder-only architectures
  - Encoder-decoder architectures
  - Tied vs untied embeddings

- **Tokenization**
  - Word-level, character-level tokenization
  - Byte-Pair Encoding (BPE)
  - Byte-level BPE
  - WordPiece
  - SentencePiece
  - Unigram language model tokenization
  - Tiktoken
  - Vocabulary size tradeoffs
  - Special tokens and chat templates
  - Tokenizer training

- **Specific LLM Architectures and Families**
  - Original Transformer (Vaswani et al.)
  - Encoder Models: BERT, RoBERTa, ALBERT, DistilBERT, DeBERTa (v1/v2/v3), ELECTRA, XLNet
  - Decoder Models: GPT-1/2/3/3.5/4/4o/5, GPT-J, GPT-NeoX
  - Encoder-Decoder: T5, FLAN-T5, BART, mT5, UL2
  - Open-weight LLMs: LLaMA 1/2/3/4, Mistral, Mixtral, Falcon, Qwen 2/3, DeepSeek V2/V3/R1, Gemma 1/2/3, Phi-3/4, Yi, Kimi, GLM, MiniMax, Nemotron, Command R/A
  - Closed Frontier Models: Claude (Sonnet/Opus), Gemini, ChatGPT/o-series
  - State Space Models (SSMs): S4, S5, S6, Mamba, Mamba-2
  - Attention-Free / Linear Attention: RWKV, RetNet, Linear Transformer, Performer, Linformer
  - Hybrid Architectures: Jamba, Granite 4.0, Qwen3-Next, Kimi Linear, Hymba
  - Mixture of Experts (MoE): Switch Transformer, GShard, Mixtral, DeepSeek-MoE, fine-grained experts, shared experts, routing strategies
  - Long-context Models
  - Multi-token prediction architectures

- **Pre-training Objectives**
  - Causal language modeling (next-token prediction)
  - Masked language modeling (MLM)
  - Span corruption (T5)
  - Prefix language modeling
  - Permutation language modeling
  - Fill-in-the-Middle (FIM)
  - Denoising objectives
  - Replaced token detection (ELECTRA)

- **Scaling and Emergent Behavior**
  - Kaplan scaling laws
  - Chinchilla scaling laws
  - Compute-optimal training
  - Emergent abilities and phase transitions
  - Loss-to-performance prediction
  - μP (maximal update parameterization)
  - Inference-time scaling vs pretraining scaling

- **Fine-tuning and Adaptation**
  - Full fine-tuning
  - Instruction tuning / SFT
  - Parameter-Efficient Fine-Tuning (PEFT)
    - Adapters, AdapterFusion
    - Prefix tuning
    - Prompt tuning
    - P-tuning v1/v2
    - LoRA
    - QLoRA
    - DoRA
    - VeRA, LoftQ, GaLore
    - IA³
  - Continued pretraining / domain adaptation
  - Knowledge distillation
  - Model merging (TIES, DARE, task arithmetic, SLERP)

- **Alignment and Preference Optimization**
  - RLHF pipeline (SFT → reward model → PPO)
  - Reward modeling
  - PPO for LLMs
  - Direct Preference Optimization (DPO)
  - Identity Preference Optimization (IPO)
  - Kahneman-Tversky Optimization (KTO)
  - Odds Ratio Preference Optimization (ORPO)
  - Simple Preference Optimization (SimPO)
  - Group Relative Policy Optimization (GRPO)
  - DAPO, REINFORCE++
  - RLAIF and Constitutional AI
  - Self-rewarding language models
  - Rejection sampling fine-tuning
  - Best-of-N and reranking

- **Reasoning and Inference-Time Techniques**
  - In-context learning and few-shot prompting
  - Zero-shot prompting
  - Chain-of-Thought (CoT)
  - Self-consistency
  - Tree-of-Thoughts
  - Graph-of-Thoughts
  - ReAct (reasoning + acting)
  - Reflexion / self-reflection
  - Process reward models vs outcome reward models
  - Test-time compute scaling
  - Long CoT reasoning models (o1, R1, QwQ)
  - Speculative reasoning / search-based decoding
  - Monte Carlo Tree Search with LLMs

- **Retrieval-Augmented Generation (RAG)**
  - Dense retrieval and embeddings
  - Sparse retrieval (BM25)
  - Hybrid retrieval
  - Reranking (cross-encoders, ColBERT, late interaction)
  - Chunking strategies
  - Vector databases (FAISS, Milvus, Pinecone, Weaviate, Qdrant, Chroma, pgvector)
  - Approximate nearest neighbor search (HNSW, IVF, PQ)
  - Multi-vector and ColBERT-style retrieval
  - Graph RAG
  - Agentic RAG
  - Long-context RAG vs retrieval
  - Evaluation of RAG (faithfulness, answer relevance)

- **Agents and Tool Use**
  - Function calling / tool calling
  - Model Context Protocol (MCP)
  - Agent loops (think → act → observe)
  - ReAct-style agents
  - Multi-agent systems and orchestration
  - Agent-to-Agent (A2A) protocols
  - Agent skills
  - Code execution and code interpreter agents
  - Web/browser agents
  - Computer-use / GUI agents
  - Memory systems for agents
  - Planning and task decomposition

- **Long-Context Techniques**
  - Sliding window attention
  - Sparse attention patterns (Longformer, BigBird)
  - Sliding window + global tokens
  - Ring attention
  - Blockwise / chunked attention
  - StreamingLLM and attention sinks
  - FlashAttention 1/2/3
  - FlashInfer
  - Position interpolation, NTK scaling, YaRN
  - KV cache management and compression
  - H2O, StreamingLLM, KV eviction

- **Inference Optimization**
  - Quantization
    - INT8, INT4
    - GPTQ
    - AWQ
    - GGUF/llama.cpp formats
    - SmoothQuant
    - FP8 (E4M3/E5M2)
    - MXFP8, MXFP4, NVFP4
    - Quantization-aware training
  - KV caching
  - PagedAttention (vLLM)
  - vAttention
  - Continuous batching / in-flight batching
  - Speculative decoding (draft models, EAGLE 1/2/3, Medusa, n-gram, suffix, DFlash)
  - Lookahead decoding
  - Tree attention / DeFT
  - Pruning (structured, unstructured, magnitude, movement)
  - Distillation
  - Inference Engines (vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, ExLlama, MLC-LLM, Triton Inference Server)
  - Speculative streaming
  - Disaggregated prefill/decode

- **Mechanistic Interpretability**
  - Probing classifiers
  - Activation patching and causal interventions
  - Logit lens, tuned lens
  - Circuit analysis
  - Induction heads
  - Superposition and polysemanticity
  - Sparse autoencoders (SAEs) for features
  - Transcoders
  - Feature steering and activation steering
  - Representation engineering
  - Dictionary learning

- **Safety, Red Teaming, Alignment**
  - Helpfulness, honesty, harmlessness (HHH)
  - Jailbreaks (universal, prompt injection, many-shot, GCG, PAIR)
  - Constitutional AI and constitutional classifiers
  - Adversarial robustness for LLMs
  - Refusal training
  - Specification gaming and reward hacking
  - Deceptive alignment
  - Sleeper agents
  - Watermarking LLM outputs
  - Model unlearning

---

## 6. Natural Language Processing

- **Text Preprocessing**
  - Tokenization (white-space, rule-based, subword)
  - Normalization, lowercasing, Unicode handling
  - Stop-word removal
  - Stemming (Porter, Snowball)
  - Lemmatization
  - Sentence segmentation
  - Text cleaning and noise removal
  - Language detection

- **Linguistic Foundations**
  - Morphology
  - Syntax
  - Semantics
  - Pragmatics
  - Constituency and dependency parsing

- **Representations**
  - One-hot and bag-of-words
  - TF-IDF
  - n-gram models
  - Word2Vec (CBOW, Skip-gram, negative sampling)
  - GloVe
  - FastText
  - ELMo
  - Contextual embeddings (BERT-style)
  - Sentence embeddings (Sentence-BERT, SimCSE)
  - Cross-encoders vs bi-encoders
  - Multilingual embeddings (mBERT, XLM-R, LaBSE)

- **Language Modeling**
  - n-gram language models and smoothing
  - Neural language models (Bengio)
  - Recurrent LMs
  - Transformer LMs

- **Core NLP Tasks**
  - POS tagging
  - Named Entity Recognition (NER)
  - Chunking
  - Constituency and dependency parsing
  - Coreference resolution
  - Semantic role labeling
  - Word sense disambiguation
  - Entity linking
  - Relation extraction
  - Event extraction
  - Text classification
  - Sentiment analysis and aspect-based sentiment
  - Topic modeling
  - Text similarity / paraphrase detection
  - Natural language inference (NLI)

- **Generation Tasks**
  - Machine translation (statistical, neural, transformer)
  - Summarization (extractive, abstractive)
  - Question answering (extractive, generative, open-domain)
  - Dialogue systems and conversational AI
  - Data-to-text generation
  - Style transfer

- **Information Retrieval and Search**
  - Inverted indexes
  - BM25 and probabilistic retrieval
  - Learning to rank
  - Dense retrieval (DPR)
  - ColBERT and late interaction
  - Cross-encoder reranking
  - Semantic search

- **Decoding Strategies**
  - Greedy
  - Beam search
  - Sampling (temperature, top-k, top-p/nucleus)
  - Min-p, typical, η-sampling
  - Contrastive decoding
  - Constrained decoding

---

## 7. Computer Vision

- **Image Processing Fundamentals**
  - Digital image representations
  - Color spaces (RGB, HSV, Lab, YCbCr)
  - Histograms and equalization
  - Filtering (linear, nonlinear, Gaussian, median)
  - Edge detection (Sobel, Canny, Laplacian)
  - Morphological operations (erosion, dilation, opening, closing)
  - Image pyramids
  - Fourier and wavelet transforms for images

- **Mathematical Foundations of CV**
  - Homogeneous coordinates
  - Projective and affine geometry
  - Camera models (pinhole, intrinsics, extrinsics, distortion)
  - Multiple-view geometry
  - Epipolar geometry and fundamental matrix
  - Stereo vision and triangulation
  - Structure from Motion
  - Bundle adjustment
  - Lie groups for rotations (SO(3), SE(3))

- **Classical Computer Vision**
  - Feature detection (Harris, FAST)
  - Feature descriptors (SIFT, SURF, ORB, BRIEF)
  - HOG (Histogram of Oriented Gradients)
  - Optical flow (Lucas-Kanade, Horn-Schunck, Farneback)
  - Feature matching, RANSAC
  - Image stitching and panoramas
  - Hough transforms

- **Image Classification**
  - ImageNet pipeline
  - Classical CNNs (LeNet through EfficientNet)
  - Data augmentation strategies
  - Test-time augmentation

- **Object Detection**
  - Two-stage: R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN
  - One-stage: YOLO (v1–v12), SSD, RetinaNet
  - Anchor-based vs anchor-free
  - FCOS, CenterNet
  - DETR, Deformable DETR, DINO-DETR
  - Grounding DINO, OWL-ViT (open-vocabulary)
  - NMS and Soft-NMS
  - mAP, IoU evaluation

- **Segmentation**
  - Semantic segmentation (FCN, U-Net, DeepLab v1–v3+, SegFormer)
  - Instance segmentation (Mask R-CNN, YOLACT)
  - Panoptic segmentation
  - Segment Anything (SAM, SAM 2)
  - Promptable segmentation

- **Other Vision Tasks**
  - Pose estimation (OpenPose, HRNet, ViTPose)
  - Keypoint detection
  - Face detection and recognition (FaceNet, ArcFace)
  - OCR (Tesseract, TrOCR, PaddleOCR)
  - Image super-resolution
  - Image inpainting
  - Image denoising and restoration
  - Depth estimation (monocular, stereo)
  - Image-to-image translation

- **Video Understanding**
  - Action recognition (I3D, SlowFast, TimeSformer, VideoMAE)
  - Video classification
  - Temporal action localization
  - Object tracking (SORT, DeepSORT, ByteTrack)
  - Video transformers
  - Video diffusion (Sora, Veo, Kling, etc.)

- **3D Vision**
  - Point clouds (PointNet, PointNet++, DGCNN)
  - Voxel-based methods
  - Neural Radiance Fields (NeRF, Instant-NGP, Mip-NeRF)
  - Gaussian Splatting (3DGS, 4DGS)
  - Mesh-based methods
  - SDFs (Signed Distance Fields)
  - Diffusion-based 3D generation

- **Vision Transformers**
  - ViT
  - DeiT
  - Swin Transformer (v1, v2)
  - PVT, Twins
  - CvT, CoAtNet (hybrid)
  - MAE (Masked Autoencoders)
  - BEiT
  - DINO, DINOv2, DINOv3
  - Token merging, dynamic tokens

- **Image Generation**
  - GAN-based generation (StyleGAN family)
  - Diffusion-based generation
    - DDPM, DDIM
    - Latent Diffusion Models
    - Stable Diffusion 1.5/2/XL/3/3.5
    - Imagen, DALL-E 2/3
    - FLUX, Ideogram
    - ControlNet, T2I-Adapter
    - LoRA fine-tuning of diffusion models
    - DreamBooth, Textual Inversion
    - IP-Adapter
  - Flow-matching image generators
  - Autoregressive image generation (Parti, MUSE)

- **Self-Supervised Vision**
  - Contrastive: SimCLR, MoCo v1/v2/v3
  - Non-contrastive: BYOL, SimSiam, Barlow Twins, VICReg
  - Distillation-based: DINO, DINOv2, DINOv3
  - Masked image modeling: MAE, SimMIM, BEiT
  - JEPA, I-JEPA, V-JEPA, V-JEPA 2

---

## 8. Speech and Audio

- **Audio Signal Processing**
  - Sampling, quantization, Nyquist
  - Time-domain and frequency-domain
  - Fourier Transform, FFT
  - Short-Time Fourier Transform (STFT)
  - Spectrograms and mel-spectrograms
  - MFCC features
  - Filter banks
  - Pitch and prosody features
  - Audio augmentation (SpecAugment)

- **Speech Recognition (ASR)**
  - HMM-GMM systems
  - DeepSpeech, DeepSpeech 2
  - CTC and CTC-based models
  - Listen-Attend-Spell
  - RNN-T (Transducer)
  - Conformer
  - Wav2Vec, Wav2Vec 2.0
  - HuBERT
  - Whisper / Whisper v3
  - Streaming ASR

- **Speech Synthesis (TTS)**
  - Concatenative and parametric TTS
  - WaveNet
  - Tacotron, Tacotron 2
  - FastSpeech 1/2
  - Glow-TTS, VITS
  - Diffusion TTS (Grad-TTS, NaturalSpeech)
  - VALL-E, VALL-E X
  - Voicebox, XTTS
  - Vocoders (Griffin-Lim, WaveGlow, HiFi-GAN, BigVGAN)

- **Other Audio Tasks**
  - Speaker recognition and verification
  - Speaker diarization
  - Voice conversion
  - Source separation
  - Audio classification (AudioSet, ESC)
  - Sound event detection
  - Music generation (MusicLM, MusicGen, Stable Audio, Suno)
  - Audio language models (AudioLM, AudioPaLM)
  - Neural audio codecs (SoundStream, Encodec)

---

## 9. Multimodal Learning

- **Foundations**
  - Modality encoders and projectors
  - Joint embedding spaces
  - Contrastive multimodal learning
  - Modality alignment
  - Modality fusion strategies (early, late, hybrid, cross-attention)
  - Tokenization across modalities (image patches, audio frames, video tubelets)
  - Discrete vs continuous multimodal tokens

- **Vision-Language Models**
  - CLIP, OpenCLIP, EVA-CLIP, SigLIP, SigLIP2
  - ALIGN
  - BLIP, BLIP-2, InstructBLIP
  - Flamingo, OpenFlamingo
  - LLaVA family (LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision)
  - MiniGPT-4, mPLUG-Owl
  - Kosmos-1/2
  - GPT-4V / GPT-4o
  - Gemini (1.5, 2.0, 3)
  - Qwen-VL, Qwen2-VL, Qwen2.5-VL
  - InternVL
  - Pixtral
  - Phi-3.5/4 Vision/Multimodal
  - DeepSeek-VL, Janus, Janus-Pro
  - Molmo
  - Grounding multimodal models

- **Audio-Language Models**
  - Qwen-Audio, Qwen2-Audio
  - SALMONN
  - SpeechGPT
  - LLaMA-Omni
  - GPT-4o audio modality
  - AudioPaLM

- **Video-Language Models**
  - VideoChat, Video-LLaVA, Video-ChatGPT
  - LLaMA-VID
  - Gemini long-video understanding
  - Video-LLaMA

- **Any-to-Any / Omni Models**
  - NExT-GPT
  - CoDi, CoDi-2
  - Unified-IO 2
  - GPT-4o
  - Gemini Native Multimodal
  - VITA

- **Cross-Modal Generation**
  - Text-to-image (Stable Diffusion, DALL-E, Imagen, FLUX, Ideogram)
  - Text-to-video (Sora, Veo, Kling, Runway, Pika)
  - Text-to-audio / text-to-music (AudioLDM, Stable Audio, MusicGen, Suno)
  - Text-to-speech (multimodal)
  - Text-to-3D (DreamFusion, Magic3D, GaussianDreamer)
  - Image-to-video, video-to-audio
  - ControlNet-style conditioning across modalities

- **Cross-Modal Retrieval**
  - Image-text retrieval
  - Audio-text retrieval
  - Video-text retrieval
  - Multimodal embeddings

- **Embodied / World-Model AI**
  - Vision-Language-Action (VLA) models (RT-2, OpenVLA, π0, SmolVLA)
  - Robotic foundation models
  - JEPA, V-JEPA, V-JEPA 2 world models
  - Sim-to-real transfer
  - Cosmos and physical world models
  - Diffusion policies for robotics

- **Multimodal Training Pipeline**
  - Stage 1: Modality alignment / projector training
  - Stage 2: Multimodal pretraining
  - Stage 3: Multimodal instruction tuning
  - Stage 4: Multimodal preference optimization (RLHF/DPO)
  - Multimodal data curation and filtering

---

## 10. ML Systems, MLOps, and LLM Systems

- **Distributed Training**
  - Data parallelism (DP, DDP)
  - Tensor / model parallelism (Megatron-style)
  - Pipeline parallelism (GPipe, PipeDream, 1F1B, interleaved)
  - Sequence parallelism
  - Context parallelism (Ring Attention)
  - Expert parallelism (MoE)
  - Sharded data parallelism
  - ZeRO (stages 1, 2, 3) and ZeRO-Offload
  - Fully Sharded Data Parallel (FSDP, FSDP2)
  - DeepSpeed
  - Megatron-LM, Megatron-DeepSpeed
  - 3D / 4D / 5D parallelism
  - Hybrid parallelism strategies
  - Communication primitives (all-reduce, all-gather, reduce-scatter, all-to-all)
  - NCCL, RCCL, Gloo
  - Gradient accumulation
  - Gradient compression

- **Hardware**
  - NVIDIA GPUs (A100, H100, H200, B100, B200, GB200)
  - AMD MI250/MI300/MI325
  - Google TPUs (v4, v5e, v5p, Trillium)
  - AWS Trainium / Inferentia
  - Cerebras, Groq, SambaNova
  - Apple Silicon, MLX
  - Memory hierarchies and HBM
  - NVLink, NVSwitch
  - InfiniBand, RoCE, Slingshot

- **Mixed-Precision and Numerical Training**
  - FP32 / TF32 baseline
  - FP16 with loss scaling
  - BF16
  - FP8 (Hopper, Blackwell)
  - Microscaling formats (MXFP8, MXFP4, NVFP4)
  - Stochastic rounding
  - Gradient scaling

- **Memory Optimization**
  - Activation checkpointing / recomputation
  - Selective activation recomputation
  - CPU and NVMe offloading
  - Optimizer state sharding
  - Paged optimizers
  - Memory-efficient attention

- **Data Engineering for LLMs**
  - Web crawl pipelines (Common Crawl, FineWeb, FineWeb-Edu, RedPajama, Dolma)
  - Deduplication (exact, fuzzy, MinHash, SimHash)
  - Quality filtering (heuristic, model-based)
  - Toxicity and PII filtering
  - Language identification
  - Document classification
  - Data mixing and domain weighting
  - Curriculum / data scheduling
  - Tokenized dataset packing
  - Streaming datasets, WebDataset, MosaicML Streaming

- **Model Serving and Inference Systems**
  - vLLM
  - SGLang
  - TensorRT-LLM
  - Text Generation Inference (TGI)
  - llama.cpp / MLX / Ollama
  - Triton Inference Server
  - TorchServe
  - Ray Serve / BentoML
  - Continuous batching engines
  - Disaggregated serving
  - Multi-LoRA / multi-tenant serving
  - Edge and on-device inference (CoreML, ONNX Runtime, TensorRT, TFLite, MediaPipe)

- **MLOps**
  - Feature stores (Feast, Tecton)
  - Model registries
  - Model and data versioning
  - CI/CD for ML
  - A/B testing and canary deployments
  - Shadow and offline evaluation
  - Drift detection (data, concept, prediction)
  - Monitoring and observability for LLMs (latency, cost, quality, hallucination)
  - LLMOps tools (LangSmith, Langfuse, Helicone)
  - Cost optimization for inference

- **Reliability and Operations**
  - Failure handling in large training runs
  - Checkpointing strategies
  - Elasticity and preemption
  - Network and node failure recovery
  - Determinism and reproducibility

---

## 11. Research Skills and PhD Skills

- **Reading Literature**
  - arXiv navigation and triage
  - Reading papers efficiently
  - Reproducing papers from scratch
  - Critical reading and peer review
  - Literature surveys

- **Writing and Communication**
  - LaTeX and BibTeX
  - Paper structure (intro, related work, method, experiments)
  - Figure design and visualization
  - Equation typesetting
  - Rebuttal writing
  - Poster and talk preparation

- **Conferences and Venues**
  - NeurIPS, ICML, ICLR, COLT
  - ACL, EMNLP, NAACL, EACL
  - CVPR, ICCV, ECCV, WACV
  - AAAI, IJCAI, UAI
  - Interspeech, ICASSP
  - CoRL, RSS, ICRA, IROS
  - MLSys, OSDI/SOSP for ML systems
  - Workshops and benchmark tracks
  - Datasets & Benchmarks tracks

- **Experimental Methodology**
  - Experimental design
  - Ablation studies
  - Statistical significance testing for ML
  - Confidence intervals on benchmarks
  - Reproducibility and seeds
  - Compute and energy reporting
  - Negative results
  - Pre-registration

- **Research Practice**
  - Open source contributions
  - Code releases and model cards
  - Dataset documentation (datasheets)
  - Research ethics, IRB
  - Research project management
  - Collaboration (Slack, Notion, shared infra)
  - Mentorship and advising
  - Grant writing and funding

---

## 12. Evaluation and Benchmarks

- **General LLM Benchmarks**
  - MMLU, MMLU-Pro
  - HellaSwag, ARC, WinoGrande
  - BIG-Bench, BIG-Bench Hard (BBH)
  - HELM (holistic evaluation)
  - AGIEval
  - TriviaQA, NaturalQuestions
  - LMArena / Chatbot Arena
  - AlpacaEval, MT-Bench, Arena-Hard

- **Reasoning Benchmarks**
  - GSM8K, MATH
  - AIME 2024/2025
  - GPQA / GPQA-Diamond
  - Humanity's Last Exam (HLE)
  - ARC-AGI / ARC-AGI-2
  - Olympiad-style math benchmarks

- **Coding Benchmarks**
  - HumanEval, HumanEval+
  - MBPP, MBPP+
  - LiveCodeBench
  - SWE-bench, SWE-bench Verified, SWE-bench Pro
  - BigCodeBench
  - APPS

- **Tool / Agent Benchmarks**
  - BFCL (Berkeley Function Calling Leaderboard)
  - τ-Bench, τ²-Bench
  - WebArena, VisualWebArena
  - GAIA
  - SWE-Lancer
  - GDPval

- **Long-Context Benchmarks**
  - Needle-in-a-Haystack
  - RULER
  - InfiniteBench
  - LongBench

- **Vision Benchmarks**
  - ImageNet, ImageNet-21k
  - COCO (detection, segmentation, captioning)
  - ADE20K, Cityscapes
  - LVIS
  - Kinetics, Something-Something v2
  - Open Images

- **Multimodal Benchmarks**
  - VQA, GQA
  - MMMU, MMMU-Pro
  - MMBench, SEED-Bench
  - ChartQA, DocVQA, OCR-Bench
  - Video-MME, MVBench
  - EgoSchema

- **Speech Benchmarks**
  - LibriSpeech, Common Voice
  - WER, CER metrics

- **Evaluation Metrics**
  - Perplexity, bits-per-byte
  - Accuracy, F1, precision, recall
  - BLEU, ROUGE, METEOR, chrF, COMET, BLEURT
  - mAP, IoU, Dice
  - FID, IS, KID, CLIP score
  - LPIPS, SSIM, PSNR
  - Word Error Rate (WER), MOS for TTS
  - Pass@k for code

- **Evaluation Methodologies**
  - Human evaluation and annotation
  - LLM-as-a-judge
  - Pairwise comparison and Elo ratings
  - Calibration and reliability diagrams
  - Robustness and stress tests
  - Contamination detection
  - Holistic / multi-dimensional evaluation

---

## 13. AI Safety, Ethics, Alignment

- **Alignment Theory**
  - Outer vs inner alignment
  - Mesa-optimization
  - Goal misgeneralization
  - Reward hacking and specification gaming
  - Scalable oversight
  - Debate, recursive reward modeling
  - Iterated amplification
  - Weak-to-strong generalization
  - Deceptive alignment

- **Safety Techniques**
  - RLHF / RLAIF
  - Constitutional AI and constitutional classifiers
  - Red teaming (manual, automated)
  - Adversarial training
  - Jailbreak detection and mitigation
  - Refusal training
  - Safe decoding and guardrails
  - Content moderation models

- **Bias, Fairness, and Ethics**
  - Algorithmic fairness definitions (demographic parity, equal odds, calibration)
  - Bias measurement and mitigation
  - Representation harms
  - Allocation harms
  - Toxicity, stereotype evaluation
  - Disparate impact
  - Ethics frameworks for AI

- **Privacy**
  - Differential privacy (DP-SGD)
  - Federated learning
  - Secure multi-party computation
  - Homomorphic encryption
  - Privacy attacks (membership inference, model inversion)
  - PII handling and redaction

- **Adversarial Robustness**
  - Adversarial examples (FGSM, PGD)
  - Adversarial training
  - Certified robustness
  - Backdoor attacks and data poisoning
  - Prompt injection and indirect prompt injection

- **Provenance and Watermarking**
  - LLM output watermarking
  - Image/video watermarking
  - C2PA and content credentials
  - Detection of AI-generated content

- **Governance and Policy**
  - EU AI Act
  - US executive orders and NIST AI RMF
  - Frontier Model Forum, responsible scaling policies
  - Model evaluations for dangerous capabilities (CBRN, cyber, autonomy)
  - Compute governance
  - Open vs closed model debates
  - Liability and IP issues

---

## 14. Emerging and Advanced Research Topics

- **Continuous and Implicit Models**
  - Neural ODEs
  - Neural SDEs
  - Neural CDEs
  - Implicit neural representations (SIREN, NeRF-like)
  - Deep equilibrium models

- **Graph Neural Networks**
  - Spectral GNNs (ChebNet, GCN)
  - Spatial GNNs (GraphSAGE)
  - Graph Attention Networks (GAT, GATv2)
  - Message Passing Neural Networks
  - Graph Transformers
  - Heterogeneous and temporal GNNs
  - Graph pooling
  - Expressivity and Weisfeiler-Lehman hierarchy

- **Geometric Deep Learning**
  - Group equivariance and invariance
  - SE(3)/E(3) equivariant networks
  - Tensor Field Networks, SE(3)-Transformer
  - EGNN
  - Steerable CNNs
  - Spherical CNNs
  - Clifford / geometric algebra networks
  - Manifold learning

- **Energy-Based Models**
  - Boltzmann machines, RBMs, DBMs
  - Score matching
  - Denoising score matching
  - Contrastive divergence
  - EBMs as unifying framework

- **World Models and Model-Based AI**
  - Dreamer v1/v2/v3
  - PlaNet
  - JEPA family (I-JEPA, V-JEPA, V-JEPA 2)
  - Genie / Cosmos / video-based world models
  - Neural simulators

- **Continual and Lifelong Learning**
  - Catastrophic forgetting
  - Elastic Weight Consolidation
  - Replay-based methods
  - Modular and task-conditional networks
  - Meta-continual learning

- **Causal Inference and Causal ML**
  - Structural causal models
  - Do-calculus
  - Counterfactuals
  - Potential outcomes framework
  - Causal discovery
  - Instrumental variables
  - Causal representation learning

- **Neuro-Symbolic AI**
  - Symbolic reasoning
  - Differentiable logic
  - Knowledge graphs and reasoning
  - Program induction and synthesis
  - Tool-augmented reasoning

- **Quantum Machine Learning (Awareness Level)**
  - Quantum computing basics
  - Variational quantum circuits
  - Quantum neural networks
  - Quantum advantage claims

- **AI for Science**
  - AlphaFold 1/2/3
  - Protein language models (ESM 1/2/3)
  - RoseTTAFold
  - Materials (GNoME, MatterGen)
  - Climate and weather (GraphCast, Pangu-Weather, Aurora)
  - Drug discovery and molecular generation
  - Mathematical reasoning (AlphaProof, AlphaGeometry)
  - Physics-informed neural networks (PINNs)
  - Neural operators (FNO, DeepONet)

- **Efficient and Frontier Architectures (2025–2026)**
  - State Space Models (S4, S5, Mamba, Mamba-2)
  - Hybrid attention/SSM models (Jamba, Hymba, Granite 4)
  - Linear attention revivals (RWKV-7, Linear Transformers)
  - Test-time training models
  - Diffusion language models (Mercury, Gemini Diffusion)
  - Block diffusion / DFlash
  - Energy-based and EBT-style transformers
  - Byte-level / tokenizer-free models (Byte Latent Transformer, MambaByte)
  - Multi-token prediction
  - Long reasoning models (o1, R1, QwQ)

- **Frontier Training Paradigms**
  - Self-play and self-improvement loops
  - Synthetic data generation pipelines
  - Curriculum and domain-randomized data
  - Reasoning RL with verifiable rewards
  - Process reward models
  - Search-augmented training (MCTS + LLM)

- **Open Research Frontiers**
  - Mechanistic interpretability at scale
  - Scalable oversight beyond human level
  - Agent foundations
  - Long-horizon autonomy
  - Embodied general intelligence
  - Compositional generalization
  - True continual learning
  - Energy-efficient AI

---

## 15. Building a Multimodal LLM From Scratch (Capstone Pipeline)

- **Stage 0: Project Planning**
  - Scope, modalities, target capabilities
  - Compute and budget planning
  - Hardware procurement / cloud selection

- **Stage 1: Data**
  - Text corpus collection and filtering
  - Image–text pair collection (LAION, DataComp, internal)
  - Video–text and audio–text data
  - Multilingual data
  - Synthetic data generation
  - Data deduplication, decontamination, licensing

- **Stage 2: Tokenization and Encoders**
  - Train BPE/SentencePiece tokenizer
  - Choose / train vision encoder (ViT, SigLIP)
  - Choose / train audio encoder (Whisper, wav2vec)
  - Choose / train video encoder
  - Discrete vs continuous modality tokens

- **Stage 3: Architecture Design**
  - Decide architecture family (dense Transformer, MoE, hybrid SSM)
  - Attention variant (MHA/GQA/MLA)
  - Positional encoding (RoPE/ALiBi)
  - Normalization (RMSNorm, pre-norm)
  - Activation (SwiGLU)
  - Multimodal connector / projector design (MLP, Q-Former, Perceiver Resampler)

- **Stage 4: Pretraining**
  - Distributed training stack (FSDP/DeepSpeed/Megatron)
  - Mixed precision and FP8/BF16
  - Learning rate and batch schedules
  - Loss curves and divergence diagnostics
  - Checkpointing and fault tolerance
  - Scaling-law-driven hyperparameter selection

- **Stage 5: Multimodal Alignment Stage**
  - Freeze LLM and train projector
  - Image–text contrastive alignment
  - Caption-style next-token training

- **Stage 6: Multimodal Pretraining**
  - Interleaved image-text-audio sequences
  - Long-context multimodal data

- **Stage 7: Instruction and Preference Tuning**
  - Multimodal SFT data
  - Multimodal DPO/ORPO/GRPO
  - Tool-use and agent fine-tuning
  - Reasoning fine-tuning

- **Stage 8: Evaluation**
  - Multimodal benchmarks
  - Human evaluation
  - Red teaming
  - Domain-specific evals

- **Stage 9: Inference and Deployment**
  - Quantization (FP8/INT4)
  - vLLM / TensorRT-LLM serving
  - Multimodal serving pipeline
  - Monitoring and feedback loops

- **Stage 10: Iteration**
  - Active learning data flywheel
  - Continued pretraining and alignment
  - Capability and safety evaluation

---

## 16. Recommended Course / Curriculum Anchors (Topics-as-Pointers)

- **Math & Foundations**
  - MIT 18.06 Linear Algebra
  - MIT 18.01/18.02 Calculus
  - MIT 6.041 / Stanford EE178 Probability
  - Boyd Convex Optimization (EE364a/b)
  - MIT 18.650 Statistics
  - MIT 6.436 Information Theory

- **Classical ML**
  - Stanford CS229
  - Stanford CS228 / CMU 10-708 (PGM)
  - Berkeley CS189

- **Deep Learning**
  - Stanford CS230
  - CMU 11-785 / 11-485
  - NYU DS-GA 1008
  - fast.ai Practical Deep Learning
  - DeepLearning.AI Specialization

- **NLP and LLMs**
  - Stanford CS224n
  - Stanford CS224u
  - Stanford CS336 (Language Modeling from Scratch)
  - Princeton COS 597G (LLMs)

- **Vision**
  - Stanford CS231n
  - CMU 16-720
  - Michigan EECS 498

- **Reinforcement Learning**
  - Berkeley CS285
  - DeepMind x UCL RL Lecture Series
  - Sutton & Barto book

- **Multimodal / Generative**
  - Berkeley CS294-158 Deep Unsupervised Learning
  - Stanford CS25 Transformers United
  - Stanford CS236 Deep Generative Models

- **ML Systems**
  - CMU 10-714 Deep Learning Systems
  - Stanford CS149 Parallel Computing
  - MIT 6.S965 TinyML and Efficient DL
  - CS336 systems lectures (GPUs, Triton, parallelism)

- **Reasoning, Agents, Safety**
  - Berkeley CS294 LLM Agents
  - MIT 6.S898 / Anthropic & DeepMind safety syllabi
  - ARENA program (alignment)

---

## Caveats

- This list intentionally errs on the side of completeness; not every leaf topic is required for every multimodal LLM project, but a PhD-level architect should at least be familiar with all of them.
- The "frontier" topics (e.g., GRPO/DAPO/RLVR, Mamba-2 hybrids, MXFP4 quantization, V-JEPA 2, MCP agents, DINOv3, FlashAttention-3, constitutional classifiers, rectified flow) reflect 2025–2026 state of the art and are evolving rapidly; the *concepts* will outlast the specific model names.
- Several topics are listed in multiple categories (e.g., diffusion models, attention mechanisms, RLHF/DPO) because they sit at the intersection of subfields; treat the duplication as cross-references, not redundancy.
- Course/curriculum anchors at the end are pointers to publicly available syllabi (Stanford CS229/CS230/CS231n/CS224n/CS336, Berkeley CS285, CMU 11-785, MIT 6.S191, fast.ai); offerings and numbering can change year-to-year.
- Mathematical depth required varies: a working LLM architect needs operational fluency in linear algebra, calculus, probability, optimization, and information theory; measure theory, functional analysis, and stochastic calculus are needed mainly for theoretical research (e.g., diffusion theory, generalization bounds).
- This is a *checklist*, not a study plan; a realistic ordering for someone with no ML background is roughly: Math foundations → Python/PyTorch → Classical ML → Deep Learning → Transformers/NLP → Vision/Audio → Multimodal → Distributed Systems → Alignment/Safety → Research methodology, with significant interleaving and project work throughout.