# Day 1 — Linear Algebra Foundations: Tensors, Matrix Operations, Eigenvalues, SVD

> **Roadmap position:** Day 1 of 30 (Week 1 — Math + Tools + Classical ML you can't skip).
> **Time budget:** ~7 hours.
> **Prerequisites assumed:** none beyond "you can write a Python function and run NumPy."
> **By end of day, you will be able to:** explain what a tensor *is*, do matrix
> multiplication by hand and in `einsum` notation, compute an SVD on paper for
> a 2×2 matrix, implement matmul / transpose / pseudoinverse / power-iteration
> from scratch in NumPy, and articulate why every one of these objects appears
> inside a transformer.

---

## How to study today (the order matters)

| Block | Time | What to do |
|---|---|---|
| **A. Orientation** | 15 min | Read §1 (TL;DR) and §2 (Why It Matters). Do not skip — these set the hooks everything else hangs on. |
| **B. Intuition build** | 45 min | Read §3 (Mental Models) and §4 (Intuitive Explanation). Re-read §4 if any analogy didn't click. **No formulas yet.** |
| **C. Formalism** | 75 min | Read §5 (Formal Definition) and §6 (Deep Theoretical Treatment). Have paper next to you. When a derivation appears, redo it on paper before reading the next line. |
| ─── short break ─── | 15 min | Stand up. Don't open Twitter. |
| **D. Worked examples** | 60 min | Do every example in §7 on paper *before* checking the answer. Run the code snippet. Modify it. |
| **E. Code lab** | 150 min | Do the four exercises in §10 (matmul, transpose, pseudoinverse, power-iteration). Reference solutions are collapsed — try first, peek only when stuck for >10 min. |
| **F. Connections + Mistakes** | 30 min | Read §9 and §12 to lock the ML context and avoid the standard traps. |
| **G. Self-check** | 30 min | Answer §13 questions without looking. Then check. Anything you got wrong → re-read the relevant section tomorrow morning before Day 2. |
| **H. Cool-down** | 15 min | Skim §11 (frontier) and §14 (glossary). Commit your code. |

**Total:** ~7 hours. If you have less time today, the non-negotiables are
**B, C, E, G** in that order — skip §11 and §15.

**Stop rule:** if you genuinely understand the eight items in §13 by end of
day, you've completed Day 1 successfully. If you don't, do not push to Day 2
— spend tomorrow morning on the gaps. Compounding is real; gaps in Week 1
become catastrophes in Week 3.

---

## 1. TL;DR

- A **scalar** is one number. A **vector** is an ordered list of numbers
  (a 1D array). A **matrix** is a 2D grid of numbers. A **tensor** is the
  general $n$-dimensional version. Modern ML lives in 3D and 4D tensors
  (batch × sequence × feature, or batch × channel × height × width).
- A matrix is two things at once: (i) a **table of numbers**; (ii) a
  **linear function** that takes a vector in and gives a vector out. The
  second view is the one that matters.
- **Matrix multiplication** $C = AB$ means: apply transformation $B$, then
  apply transformation $A$, to whatever vector you feed in. This is why
  attention is "$Q K^\top$" — it's a *function* turning queries into
  similarity scores.
- **Transpose** $A^\top$ swaps rows and columns. For real matrices it has
  the algebraic identity $(AB)^\top = B^\top A^\top$ — this single identity
  drives the shape arithmetic of backprop.
- **Inverse** $A^{-1}$ undoes the transformation $A$, when one exists. For
  non-square or singular matrices, the **Moore–Penrose pseudoinverse**
  $A^+$ is the closest thing to an inverse and is what *every* least-squares
  solver computes under the hood.
- **Eigenvalues / eigenvectors**: $Av = \lambda v$. The eigenvector $v$ is
  a direction $A$ stretches without rotating; $\lambda$ is the stretch
  factor. PageRank is a single eigenvector. Self-attention's repeated
  application is governed by the spectrum of its effective weight matrix.
- **SVD (Singular Value Decomposition)**: $A = U \Sigma V^\top$. Every
  matrix — square or not, real or complex — has one. It exposes the
  matrix as "rotate, stretch along axes, rotate again." It is the most
  important matrix factorization in ML, period. It underpins PCA, LoRA,
  low-rank adapters, model compression, and the geometry of attention.
- **Tensors generalize all of this** to arbitrary rank. PyTorch's `Tensor`
  is just a multi-dimensional array with autograd attached. `einsum` is
  the universal language for tensor contractions and is how to read modern
  paper implementations.
- **Practical consequence for the next 29 days**: you will see matmul on
  every page of every paper. SVD will appear by Day 7 (PCA), Day 13 (MHA),
  Day 19 (LoRA). Lock these in *today*.

---

## 2. Why It Matters

A modern transformer forward pass is, end to end, a sequence of matrix
multiplications interleaved with cheap element-wise nonlinearities. If you
open any LLM inference engine — vLLM, llama.cpp, TensorRT-LLM — the hot
path is `gemm` calls (general matrix multiply). Concretely:

- **Embedding lookup** = indexing a $V \times d$ matrix (vocab × hidden).
- **Self-attention** $\mathrm{softmax}(QK^\top / \sqrt{d_k}) V$ is three
  matmuls and one softmax.
- **Feed-forward block** is two matmuls: $\sigma(xW_1) W_2$.
- **Output head** is one matmul plus softmax: $h W_o$.

So a transformer is essentially: matmul, matmul, softmax, matmul, matmul,
add, matmul, matmul, softmax, matmul, … for tens of billions of parameters,
billions of times per second on H100s. The reason GPUs exist *as we know
them* is to do matrix multiplication fast. Tensor Cores on H100 do nothing
else.

Beyond the forward pass:

- **Backprop** is the chain rule expressed as matrix products. Every gradient
  you'll ever compute is a Jacobian times a vector (vector-Jacobian product
  in PyTorch's autograd).
- **PCA** (Phase 3) is "take the top-$k$ singular vectors."
- **LoRA** (Day 19) approximates a weight update as $\Delta W = BA$ where
  $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $r \ll d$.
  That's literally a rank-$r$ approximation — the SVD truncation result.
- **Quantization** (Day 25) decides which singular directions to preserve
  when storing weights in 4 bits.
- **Diffusion models** (Day 10) use the eigenvalues of the noise covariance
  to derive the noise schedule.
- **PageRank** (Google's original algorithm) is the dominant eigenvector
  of the web's link matrix.

Without linear algebra you can run `model.generate(prompt)`. With linear
algebra you can debug it, modify it, build it, and understand papers about
it. That is the entire difference between an API user and an LLM architect.

**One sentence summary:** linear algebra is not "math you need for ML" —
linear algebra *is* the substrate ML runs on, the way C is the substrate
your kernel runs on. Today is the day you stop being a tourist about it.

---

## 3. Prerequisites and Mental Models

### What you need to recall

- **Basic arithmetic and algebra.** That's it. If you can solve
  $3x + 4 = 19$ for $x$, you have enough.
- **Functions.** A function $f: A \to B$ takes inputs from set $A$ and
  produces outputs in set $B$. We'll be doing this with $A = \mathbb{R}^n$
  and $B = \mathbb{R}^m$.
- **Python + NumPy basics.** `import numpy as np`, `np.array([1,2,3])`,
  `A @ B` for matmul, `A.shape`, `A.T` for transpose. If any of this is
  unfamiliar, spend 30 minutes on the NumPy quickstart before continuing.

### The four mental models you must internalize today

**Mental Model 1 — Vectors are arrows; matrices are arrow-mappers.**
A vector $v \in \mathbb{R}^2$ is an arrow from the origin to the point
$(v_1, v_2)$. A 2×2 matrix is a *machine* that eats an arrow and produces
another arrow. The machine has two settings: where does it send the unit
$x$-arrow $(1,0)$, and where does it send the unit $y$-arrow $(0,1)$?
Those two destinations are the **columns** of the matrix. *Once you know
where the basis arrows go, you know where every arrow goes,* because the
machine is linear: $A(2 \vec x + 3 \vec y) = 2 A\vec x + 3 A \vec y$.

This is the single most important sentence in this document. Re-read it.

**Mental Model 2 — A matrix is a function as a data structure.**
You're a backend engineer. You're used to functions and data structures
being different things. In linear algebra they collapse. A matrix is
*simultaneously*:

- a 2D array of numbers you can index (`A[i][j]`), and
- a callable object you can apply to a vector (`A @ v`), and
- a composable transformation (`A @ B` = "apply B then A").

Think of a matrix as a **serialized representation of a linear function**.
`@` is the deserialization step that actually runs it. Matrix multiplication
`A @ B` is *function composition* serialized in turn — the resulting array
is the matrix of `(v -> A(B(v)))`.

**Mental Model 3 — Eigenvectors are fixed directions.**
Most arrows, when fed into a matrix machine, come out pointing somewhere
different. Some arrows come out pointing in the *same direction* (possibly
longer, shorter, or flipped). Those special arrows are eigenvectors.
The scalar saying how much they got stretched is the eigenvalue.

Mechanical analogy: take a rubber sheet, draw arrows on it, then stretch
the sheet in some weird way (e.g., 2× horizontally, 0.5× vertically, plus
a 30° rotation). Most drawn arrows now point in new directions. But there
are special arrows — the ones aligned with the principal stretch axes —
that still point the same way they did before, just longer or shorter.
Those are the eigenvectors. The stretch factors are the eigenvalues.

**Mental Model 4 — SVD is "every matrix is three simple matrices."**
Any rectangular matrix $A$ can be written as $A = U \Sigma V^\top$ where:

- $V^\top$ is a **rotation** (of the input space),
- $\Sigma$ is a **stretch** along each axis (a diagonal scaling),
- $U$ is a **rotation** (of the output space).

So *every* linear transformation, no matter how complicated it looks, is
just: rotate, stretch, rotate. This is the deepest single fact in linear
algebra and the most useful one in ML.

### Misconceptions to pre-empt now

- **"Tensors are just multi-dimensional arrays."** True in PyTorch /
  NumPy. False in physics and mathematics, where "tensor" has a coordinate-
  independence requirement. **For ML purposes the array definition is fine
  and we'll use it**, but you should know the distinction so papers about
  geometric deep learning (Phase 8) don't confuse you.
- **"$AB$ and $BA$ are the same thing."** Almost never. Matrix
  multiplication is **not** commutative. Function composition isn't
  either: "put on socks then shoes" ≠ "put on shoes then socks."
- **"Every matrix has an inverse."** No. Only square matrices with
  non-zero determinant (equivalently: full rank, equivalently: all
  non-zero singular values). Non-square matrices never have inverses
  — they have pseudoinverses.
- **"Eigenvalues are real."** No, in general they're complex. They're
  real for symmetric matrices (a huge class in ML, including all
  covariance matrices and all $A^\top A$ products).
- **"SVD is some advanced thing."** No. It's the most fundamental
  factorization. Treat it as the *default* way to understand a matrix.

---

## 4. Intuitive Explanation (No Math Yet)

Let's walk through the whole topic without writing a single formula.

### Scalars, vectors, matrices, tensors — the data ladder

Imagine you're building a recommendation system:

- A single user's age is a **scalar**: `35`.
- That user's full profile (age, income, num_purchases, avg_session_min)
  is a **vector**: `[35, 72000, 47, 18.5]`. It's an ordered list of 4
  numbers — a point in 4-dimensional space.
- A whole table of users' profiles (1000 users × 4 features) is a
  **matrix**: 1000 rows, 4 columns. Two indices needed: row $i$ and
  column $j$.
- The history of those tables over the last 30 days (30 days × 1000 users
  × 4 features) is a **3D tensor**. Three indices: day, user, feature.
- Stack RGB images of these users (batch × height × width × 3) and you
  have a 4D tensor. Add a time dimension for video and you have 5D.

So a tensor is just the natural generalization: "how many indices do I
need to address one number?" The answer is the **rank** (or "order", or
"number of dimensions"). Confusingly, "rank" also has a different meaning
for matrices — see §6.

In an LLM, the canonical tensor shape is `(batch, sequence, hidden_dim)`
— a 3D tensor. For a batch of 8 prompts of length 1024 with hidden size
4096, that's a `(8, 1024, 4096)` tensor holding about 33 million numbers.

### What does a matrix *do*?

A matrix takes a vector and produces another vector. Concrete example:
a 2×2 matrix takes a 2D arrow and gives back a 2D arrow. The matrix

```
[ 2  0 ]
[ 0  3 ]
```

doubles the $x$-coordinate and triples the $y$-coordinate. It's a stretch.

The matrix

```
[ 0  -1 ]
[ 1   0 ]
```

takes any arrow and rotates it 90° counter-clockwise. (Try it on
$(1,0)$ — you get $(0,1)$. Try it on $(0,1)$ — you get $(-1,0)$.)

That's the entire game. Every linear function on vectors corresponds to
exactly one matrix, and vice versa. The columns of the matrix are
*literally* where the basis vectors get sent.

### Matrix multiplication as "do this, then do that"

If $A$ is "rotate 90°" and $B$ is "scale by 2," then $AB$ is "scale by 2,
then rotate 90°." The product matrix is what you get if you compose the
two transformations into one. Order matters: $BA$ would be "rotate 90°,
then scale by 2," which happens to be the same answer here but isn't in
general.

### Inverse: undo

If a matrix $A$ rotates by 90°, then $A^{-1}$ rotates by -90°. If $A$
doubles everything, $A^{-1}$ halves everything. If $A$ projects all
arrows onto a line (squashing 2D to 1D), then $A^{-1}$ **does not
exist** — once you've squashed, you can't unsquash, because you've lost
information. Whether $A^{-1}$ exists is a question about information
preservation.

### Pseudoinverse: the best you can do when inverse doesn't exist

If $A$ is non-square or singular, we can't invert. But we can ask: "given
$Ax = b$, find the $x$ that makes $Ax$ as close to $b$ as possible, and
among all such $x$'s, pick the smallest one." The function that returns
that $x$ is the pseudoinverse $A^+$. This is what `np.linalg.lstsq` does.
It's how you solve any over- or under-determined system.

### Eigenvectors: directions the matrix respects

Apply a matrix $A$ to a random vector. The result usually points somewhere
else. But there are typically a few special vectors — the eigenvectors —
that come back pointing the same way, just scaled. Those special directions
tell you the "personality" of $A$: what it stretches, what it shrinks,
what it fixes, what it flips.

In PageRank, the web is a graph; "applying the link matrix" means "one
step of a random surfer." The eigenvector with eigenvalue 1 is the
long-run probability distribution of where the surfer ends up. That
vector is the PageRank score for every page.

### SVD: the universal "rotate, stretch, rotate" decomposition

Take any matrix $A$. There's a way to express what it does in three steps:

1. **Rotate** the input so that "the directions $A$ cares about" line up
   with the standard axes. (This is $V^\top$.)
2. **Stretch** each axis independently by some non-negative amount.
   (This is $\Sigma$, the singular values.)
3. **Rotate** to produce the final output orientation. (This is $U$.)

The singular values $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ in $\Sigma$
tell you how much "energy" the matrix has along each principal axis. If
$\sigma_1$ is huge and the rest are tiny, $A$ is effectively a rank-1
transformation — it squashes almost everything down to one direction.
That's the entire mathematical foundation of LoRA, model compression,
PCA, denoising, and dozens of other techniques.

Pause and re-read this section if any of the four pictures (data ladder,
matrix-as-function, eigenvectors-as-fixed-directions, SVD-as-rotate-
stretch-rotate) isn't yet solid. The math in the next section will only
land if these intuitions are in place.

---

## 5. Formal Definition and Notation

We work over the real numbers $\mathbb{R}$ throughout. Everything
generalizes to complex numbers $\mathbb{C}$ with minor tweaks (transpose
becomes conjugate transpose, denoted $A^*$ or $A^H$).

### Scalars, vectors, matrices, tensors

- **Scalar:** $x \in \mathbb{R}$. Lowercase italic.
- **Vector:** $\mathbf{v} \in \mathbb{R}^n$ — an $n$-tuple of real numbers
  $(v_1, v_2, \ldots, v_n)$. Bold lowercase. By convention vectors are
  *column vectors* (shape $n \times 1$); the row-vector form is the
  transpose $\mathbf{v}^\top$ (shape $1 \times n$).
- **Matrix:** $A \in \mathbb{R}^{m \times n}$ — $m$ rows, $n$ columns.
  Capital italic. Entry in row $i$, column $j$ is $A_{ij}$ or $a_{ij}$.
- **Tensor:** $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}$.
  Calligraphic capital, or just "a rank-$k$ tensor of shape
  $(d_1, \ldots, d_k)$." Element accessed as $\mathcal{T}_{i_1 i_2 \ldots i_k}$.

**Notational warning:** the literature is inconsistent. Some textbooks
(Goodfellow et al.) use bold lowercase for vectors and bold uppercase for
matrices. Some papers don't bold anything. ML papers use "tensor" for any
multi-dim array; physics/math uses "tensor" for a coordinate-independent
geometric object. We use the ML convention.

### Matrix as linear map

A matrix $A \in \mathbb{R}^{m \times n}$ defines a function
$f_A : \mathbb{R}^n \to \mathbb{R}^m$ by

$$
f_A(\mathbf{x}) = A \mathbf{x}, \quad \text{where} \quad (A\mathbf{x})_i = \sum_{j=1}^{n} A_{ij} x_j. \tag{1}
$$

This function is **linear**, meaning it satisfies, for all
$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ and all $\alpha, \beta \in \mathbb{R}$:

$$
f_A(\alpha \mathbf{x} + \beta \mathbf{y}) = \alpha f_A(\mathbf{x}) + \beta f_A(\mathbf{y}). \tag{2}
$$

**Equivalent definitions** of "linear function from $\mathbb{R}^n$ to
$\mathbb{R}^m$":

1. Satisfies $(2)$.
2. Can be written as $A\mathbf{x}$ for some unique matrix $A$.
3. Sends the zero vector to the zero vector and is "additive and
   homogeneous."

These are the same object viewed three ways. Engineers find (2) clearest;
mathematicians prefer (1); computational people use (3).

### Matrix multiplication

For $A \in \mathbb{R}^{m \times p}$ and $B \in \mathbb{R}^{p \times n}$,
the product $C = AB \in \mathbb{R}^{m \times n}$ is defined by

$$
C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}. \tag{3}
$$

Three equivalent ways to read this:

- **Entry-wise** (Equation 3): each output entry is the dot product of
  a row of $A$ with a column of $B$.
- **Function composition**: $f_{AB}(\mathbf{x}) = f_A(f_B(\mathbf{x}))$.
- **Column view**: column $j$ of $AB$ is $A$ times column $j$ of $B$.

The inner dimensions must match ($A$ is $m \times p$, $B$ is $p \times n$).
The outer dimensions become the output shape ($m \times n$).

### Transpose

For $A \in \mathbb{R}^{m \times n}$, the **transpose**
$A^\top \in \mathbb{R}^{n \times m}$ is defined by $(A^\top)_{ij} = A_{ji}$.

Key identities (all derivable from the entry definitions):

$$
(A^\top)^\top = A, \quad (A + B)^\top = A^\top + B^\top, \quad (AB)^\top = B^\top A^\top. \tag{4}
$$

The last one is the one you'll use constantly. **Note the order
reversal.** Proof in §6.

### Inverse and pseudoinverse

A square matrix $A \in \mathbb{R}^{n \times n}$ is **invertible** if there
exists $A^{-1} \in \mathbb{R}^{n \times n}$ with

$$
A A^{-1} = A^{-1} A = I_n, \tag{5}
$$

where $I_n$ is the $n \times n$ identity matrix (1s on diagonal, 0s
elsewhere). $A$ is invertible iff its determinant is non-zero, iff its
columns are linearly independent, iff its rank is $n$, iff 0 is not an
eigenvalue — these are all the same condition.

The **Moore–Penrose pseudoinverse** $A^+ \in \mathbb{R}^{n \times m}$ of
*any* matrix $A \in \mathbb{R}^{m \times n}$ is the unique matrix
satisfying the four Penrose conditions:

$$
A A^+ A = A, \quad A^+ A A^+ = A^+, \quad (AA^+)^\top = AA^+, \quad (A^+A)^\top = A^+A. \tag{6}
$$

When $A$ is square and invertible, $A^+ = A^{-1}$. Otherwise, $A^+$ is
the next best thing — see §6 for what it solves.

### Determinant and trace

The **trace** $\mathrm{tr}(A)$ of a square matrix is the sum of its
diagonal entries:

$$
\mathrm{tr}(A) = \sum_{i=1}^n A_{ii}. \tag{7}
$$

The **determinant** $\det(A)$ is a single number that measures the
signed volume scaling of the transformation $A$. For a 2×2 matrix:

$$
\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc. \tag{8}
$$

If $\det(A) = 0$, the matrix squashes space to a lower dimension (and is
not invertible).

### Eigenvalues and eigenvectors

A non-zero vector $\mathbf{v} \in \mathbb{R}^n$ is an **eigenvector** of
$A \in \mathbb{R}^{n \times n}$ with **eigenvalue** $\lambda \in \mathbb{C}$
if

$$
A \mathbf{v} = \lambda \mathbf{v}. \tag{9}
$$

The set of all eigenvalues of $A$ is its **spectrum**. They are the roots
of the **characteristic polynomial** $\det(A - \lambda I) = 0$, which is
degree $n$ and hence has $n$ roots (counted with multiplicity) in $\mathbb{C}$.

### Singular Value Decomposition (SVD)

For *any* matrix $A \in \mathbb{R}^{m \times n}$ there exist orthogonal
matrices $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$
(an orthogonal matrix $Q$ satisfies $Q^\top Q = QQ^\top = I$, equivalently
its columns are orthonormal) and a "diagonal" rectangular matrix
$\Sigma \in \mathbb{R}^{m \times n}$ with non-negative entries
$\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{\min(m,n)} \ge 0$ on the
diagonal and zeros elsewhere, such that

$$
A = U \Sigma V^\top. \tag{10}
$$

The $\sigma_i$ are the **singular values** of $A$. The columns of $U$
are **left singular vectors**; the columns of $V$ are **right singular
vectors**. The decomposition exists, is essentially unique (up to sign
ambiguities and ordering), and is the workhorse of computational linear
algebra.

The number of non-zero singular values equals the **rank** of $A$.

---

## 6. Deep Theoretical Treatment

### 6.1 The four interpretations of $A\mathbf{x}$

A single matrix-vector product can be read four ways. *All four are true
simultaneously.* You should be able to switch between them fluidly.

**(i) Row view.** $(A\mathbf{x})_i$ is the dot product of row $i$ of $A$
with $\mathbf{x}$. This is what you compute by hand.

**(ii) Column view.** $A\mathbf{x}$ is a **linear combination of the
columns of $A$**, with coefficients $x_1, x_2, \ldots, x_n$:

$$
A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n, \tag{11}
$$

where $\mathbf{a}_j$ is the $j$-th column of $A$. **This is the most
important view.** It says: the set of all possible outputs $A\mathbf{x}$
is exactly the span of the columns of $A$ (a subspace of $\mathbb{R}^m$
called the **column space** or **range** of $A$).

**(iii) Function view.** $\mathbf{x} \mapsto A\mathbf{x}$ is a linear
function from $\mathbb{R}^n$ to $\mathbb{R}^m$.

**(iv) System view.** $A\mathbf{x} = \mathbf{b}$ is a system of $m$
linear equations in $n$ unknowns. Solving the system means inverting the
function.

### 6.2 Why matrix multiplication is defined that way

The formula $(AB)_{ij} = \sum_k A_{ik} B_{kj}$ looks ugly until you see
where it comes from. **Derivation:**

Let $f_A : \mathbb{R}^p \to \mathbb{R}^m$ and $f_B : \mathbb{R}^n \to \mathbb{R}^p$.
Define $g = f_A \circ f_B$, so $g(\mathbf{x}) = f_A(f_B(\mathbf{x}))$.
We want to find the matrix $C$ such that $g(\mathbf{x}) = C\mathbf{x}$.

By definition,

$$
g(\mathbf{x})_i = (A (B\mathbf{x}))_i = \sum_{k=1}^p A_{ik} (B\mathbf{x})_k = \sum_{k=1}^p A_{ik} \sum_{j=1}^n B_{kj} x_j.
$$

Swap the order of summation:

$$
g(\mathbf{x})_i = \sum_{j=1}^n \left( \sum_{k=1}^p A_{ik} B_{kj} \right) x_j.
$$

Compare to $g(\mathbf{x})_i = \sum_j C_{ij} x_j$. We get

$$
C_{ij} = \sum_{k=1}^p A_{ik} B_{kj},
$$

which is exactly equation (3). **Matrix multiplication is defined the way
it is precisely because we want $(AB)\mathbf{x} = A(B\mathbf{x})$ to
work**, i.e., we want the product matrix to represent the composition of
the two linear functions. Every other property of matmul (associativity,
distributivity, non-commutativity) follows from function composition.

### 6.3 Proof of $(AB)^\top = B^\top A^\top$

Compute the $(i, j)$-entry of each side.

LHS: $((AB)^\top)_{ij} = (AB)_{ji} = \sum_k A_{jk} B_{ki}$.

RHS: $(B^\top A^\top)_{ij} = \sum_k (B^\top)_{ik} (A^\top)_{kj} = \sum_k B_{ki} A_{jk}$.

Multiplication of scalars commutes, so $A_{jk} B_{ki} = B_{ki} A_{jk}$,
and the two sums are equal. $\blacksquare$

This identity is why, in backprop, the gradient of a matmul with respect
to its input involves the transpose of the weight matrix. Specifically,
if $\mathbf{y} = W\mathbf{x}$ and you know $\partial L / \partial \mathbf{y}$,
then $\partial L / \partial \mathbf{x} = W^\top (\partial L / \partial \mathbf{y})$.
The transpose isn't an arbitrary trick — it falls out of the chain rule
plus equation (4).

### 6.4 Rank, null space, range — the four fundamental subspaces

For $A \in \mathbb{R}^{m \times n}$:

- **Column space** $\mathcal{C}(A) \subseteq \mathbb{R}^m$: all vectors
  expressible as $A\mathbf{x}$. Dimension = **rank** of $A$.
- **Null space** $\mathcal{N}(A) \subseteq \mathbb{R}^n$: all $\mathbf{x}$
  with $A\mathbf{x} = \mathbf{0}$. Dimension = $n - \mathrm{rank}(A)$.
- **Row space** $\mathcal{C}(A^\top) \subseteq \mathbb{R}^n$: column space
  of the transpose. Same dimension as column space (= rank).
- **Left null space** $\mathcal{N}(A^\top) \subseteq \mathbb{R}^m$: null
  space of the transpose. Dimension = $m - \mathrm{rank}(A)$.

These satisfy the **rank-nullity theorem**:

$$
\mathrm{rank}(A) + \mathrm{dim}\,\mathcal{N}(A) = n. \tag{12}
$$

Intuition: $A$ takes an $n$-dimensional input space, kills off the null
space (dim $n - r$), and embeds what's left into the column space (dim $r$).
No dimension is created from nothing; whatever isn't preserved is destroyed.

This is why low-rank matrices compress: a rank-$r$ matrix can be
factored as $A = BC^\top$ where $B \in \mathbb{R}^{m \times r}$ and
$C \in \mathbb{R}^{n \times r}$, costing $r(m+n)$ numbers to store instead
of $mn$. LoRA is precisely this trick applied to weight updates.

### 6.5 Eigendecomposition

For a square $A \in \mathbb{R}^{n \times n}$, the **characteristic
polynomial** $p(\lambda) = \det(A - \lambda I)$ has $n$ roots
$\lambda_1, \ldots, \lambda_n \in \mathbb{C}$ counted with multiplicity.
For each $\lambda_i$, the equation $(A - \lambda_i I)\mathbf{v} = \mathbf{0}$
has at least one non-trivial solution $\mathbf{v}_i$ (the eigenvector).

If $A$ has $n$ linearly independent eigenvectors, stack them into a matrix
$P$ (columns = eigenvectors) and put the eigenvalues on the diagonal of a
matrix $D$. Then

$$
A = PDP^{-1}. \tag{13}
$$

This is the **eigendecomposition**. It works iff $A$ is **diagonalizable**.
Not all matrices are. The shear

$$
A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
$$

has both eigenvalues equal to 1 but only one eigenvector direction —
it's not diagonalizable. Such matrices need the Jordan form.

**Spectral theorem (the most important special case).** If $A$ is real
and symmetric ($A = A^\top$), then $A$ has $n$ real eigenvalues and an
orthonormal basis of eigenvectors. Equivalently,

$$
A = Q \Lambda Q^\top, \quad Q^\top Q = I, \quad \Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n). \tag{14}
$$

This is gold. **Every covariance matrix is symmetric and positive
semidefinite, so equation (14) applies to it.** PCA, Gaussian processes,
and a thousand other ML constructions live inside this theorem.

### 6.6 Why SVD exists for every matrix

The eigendecomposition fails for non-square matrices and for non-
diagonalizable square ones. The SVD never fails. Here is the construction.

Given $A \in \mathbb{R}^{m \times n}$, consider $A^\top A \in \mathbb{R}^{n \times n}$.

**Claim 1:** $A^\top A$ is symmetric and positive semidefinite (PSD).

*Proof of symmetric:* $(A^\top A)^\top = A^\top (A^\top)^\top = A^\top A$. $\checkmark$

*Proof of PSD:* For any $\mathbf{x}$, $\mathbf{x}^\top (A^\top A) \mathbf{x} = (A\mathbf{x})^\top (A\mathbf{x}) = \|A\mathbf{x}\|^2 \ge 0$. $\checkmark$

By the spectral theorem (eq. 14), $A^\top A = V \Lambda V^\top$ with
$\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$, all $\lambda_i \ge 0$,
ordered $\lambda_1 \ge \cdots \ge \lambda_n \ge 0$.

Define the **singular values** $\sigma_i = \sqrt{\lambda_i}$.

For each $i$ with $\sigma_i > 0$, define $\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$.

**Claim 2:** The $\mathbf{u}_i$ are orthonormal.

$\mathbf{u}_i^\top \mathbf{u}_j = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^\top A^\top A \mathbf{v}_j = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^\top (\lambda_j \mathbf{v}_j) = \frac{\sigma_j^2}{\sigma_i \sigma_j} \mathbf{v}_i^\top \mathbf{v}_j = \frac{\sigma_j}{\sigma_i} \delta_{ij}$,

which is $0$ for $i \ne j$ and $1$ for $i = j$. $\checkmark$

(Extend $\{\mathbf{u}_i\}$ to an orthonormal basis of $\mathbb{R}^m$ via
Gram–Schmidt to handle the $\sigma_i = 0$ case.)

Now stack the $\mathbf{u}_i$ as columns of $U$ and the $\mathbf{v}_i$ as
columns of $V$, with $\sigma_i$ on the diagonal of $\Sigma$. By construction,
$A \mathbf{v}_i = \sigma_i \mathbf{u}_i$ for each $i$, which is exactly
$AV = U\Sigma$, i.e., $A = U\Sigma V^\top$. $\blacksquare$

**Consequence — the Eckart–Young theorem:** the best rank-$k$
approximation to $A$ in Frobenius or spectral norm is obtained by
keeping the top $k$ singular values and zeroing the rest:

$$
A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top. \tag{15}
$$

This is the theoretical foundation of LoRA, of image compression, of
PCA dimensionality reduction, and of model distillation via low-rank
weight surgery. **Eckart–Young is one of the most useful theorems in ML.**

### 6.7 Pseudoinverse via SVD

Given $A = U\Sigma V^\top$, the pseudoinverse is

$$
A^+ = V \Sigma^+ U^\top, \tag{16}
$$

where $\Sigma^+$ is obtained from $\Sigma$ by transposing it and replacing
each non-zero diagonal entry $\sigma_i$ with $1/\sigma_i$. (Zero entries
stay zero.) This always exists and is computable.

The least-squares problem $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|^2$
has solution $\mathbf{x}^* = A^+ \mathbf{b}$. When $A$ is tall and full-rank
(overdetermined), this is the unique minimizer. When $A$ is wide
(underdetermined), $A^+ \mathbf{b}$ is the *minimum-norm* solution among
infinitely many.

### 6.8 Edge cases, failure modes, numerical stability

- **Singular matrices.** Don't call `np.linalg.inv` on a near-singular
  matrix; the result will be garbage scaled by a huge number. Use
  `np.linalg.lstsq` (pseudoinverse / least squares) instead.
- **Condition number.** $\kappa(A) = \sigma_1 / \sigma_n$. If
  $\kappa = 10^k$, you lose about $k$ digits of precision when solving
  $A\mathbf{x} = \mathbf{b}$. In float32 (~7 digits), $\kappa > 10^6$ is
  alarming; in float16 (~3 digits), $\kappa > 10^2$ is alarming. This is
  why mixed-precision training (Day 25) needs gradient scaling and loss
  scaling.
- **`@` vs `*` in NumPy.** `A @ B` is matrix multiplication. `A * B` is
  **element-wise** (Hadamard) product, which requires the shapes to
  match. New PyTorch users mix these up constantly.
- **Broadcasting.** NumPy/PyTorch silently broadcast shapes that "look
  compatible." This is a feature for productivity and a footgun for
  correctness. Always `print(x.shape)` when in doubt.
- **Non-uniqueness of SVD.** Sign of singular vectors is ambiguous; if
  $A = U\Sigma V^\top$, then $A = (-U_{:,1} \,|\, U_{:,2:}) \Sigma (-V_{:,1} \,|\, V_{:,2:})^\top$
  also works. Don't compare SVDs entry-by-entry across implementations.
- **Eigendecomposition can be complex even for real $A$.** If you call
  `np.linalg.eig` on a non-symmetric real matrix and get complex
  numbers, that's correct, not a bug.

### 6.9 Complexity

| Operation | Cost (naive) | Cost (best known) | Practical library |
|---|---|---|---|
| Matvec $A\mathbf{x}$, $A \in \mathbb{R}^{m \times n}$ | $O(mn)$ | $O(mn)$ | BLAS gemv |
| Matmul $AB$, both $n \times n$ | $O(n^3)$ | $O(n^{2.371})$ (Strassen, Coppersmith–Winograd; not used in practice) | BLAS gemm (Strassen sometimes for huge $n$) |
| Transpose | $O(mn)$ | $O(mn)$, often $O(1)$ if just a view | NumPy stride trick |
| Inverse (Gaussian elimination) | $O(n^3)$ | $O(n^{2.371})$ theoretical | LAPACK `getrf` + `getri` |
| Pseudoinverse via SVD | $O(\min(m,n)^2 \max(m,n))$ | same | LAPACK `gesdd` |
| Eigendecomp (symmetric) | $O(n^3)$ | $O(n^3)$ | LAPACK `syevd` |
| SVD | $O(\min(m,n)^2 \max(m,n))$ | same | LAPACK `gesdd` |
| Power iteration (top eigenvector) | $O(n^2)$ per iteration | converges in ~$\log(1/\epsilon) / \log(\lambda_1/\lambda_2)$ iters | hand-rolled |

The $O(n^3)$ scaling for full decomposition is why we rarely SVD a
billion-parameter weight matrix exactly. Truncated SVD via randomized
algorithms (Halko–Martinsson–Tropp 2011) reduces this to roughly
$O(mn \cdot r)$ when only the top $r$ singular values are needed.

---

## 7. Worked Examples

### Example 1 (easy) — Matrix-vector product by hand

Compute $A\mathbf{x}$ where

$$
A = \begin{pmatrix} 2 & 1 \\ 0 & 3 \\ 1 & -1 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} 4 \\ 5 \end{pmatrix}.
$$

**Row view (mechanical).** Take dot products of each row with $\mathbf{x}$:

- Row 1: $2 \cdot 4 + 1 \cdot 5 = 13$.
- Row 2: $0 \cdot 4 + 3 \cdot 5 = 15$.
- Row 3: $1 \cdot 4 + (-1) \cdot 5 = -1$.

So $A\mathbf{x} = (13, 15, -1)^\top$.

**Column view (conceptual).** $A\mathbf{x}$ is $4$ times column 1 plus
$5$ times column 2:

$$
4 \begin{pmatrix} 2 \\ 0 \\ 1 \end{pmatrix} + 5 \begin{pmatrix} 1 \\ 3 \\ -1 \end{pmatrix} = \begin{pmatrix} 8 \\ 0 \\ 4 \end{pmatrix} + \begin{pmatrix} 5 \\ 15 \\ -5 \end{pmatrix} = \begin{pmatrix} 13 \\ 15 \\ -1 \end{pmatrix}.
$$

Same answer; different mental model. Get fluent at switching.

### Example 2 (medium) — Eigendecomposition of a 2×2 by hand

Find the eigenvalues and eigenvectors of

$$
A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}.
$$

**Step 1: characteristic polynomial.**

$$
\det(A - \lambda I) = \det \begin{pmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{pmatrix} = (2-\lambda)^2 - 1.
$$

Set to zero: $(2-\lambda)^2 = 1$, so $2 - \lambda = \pm 1$, giving
$\lambda_1 = 3$, $\lambda_2 = 1$.

**Step 2: eigenvector for $\lambda_1 = 3$.** Solve $(A - 3I)\mathbf{v} = \mathbf{0}$:

$$
\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \mathbf{0}.
$$

Both rows say $v_1 = v_2$. Pick $\mathbf{v}_1 = (1, 1)^\top$ (or any
non-zero scalar multiple). Normalized: $\mathbf{v}_1 = \frac{1}{\sqrt 2}(1, 1)^\top$.

**Step 3: eigenvector for $\lambda_2 = 1$.** $(A - I)\mathbf{v} = \mathbf{0}$:

$$
\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \mathbf{v} = \mathbf{0}.
$$

Says $v_1 = -v_2$. Pick $\mathbf{v}_2 = \frac{1}{\sqrt 2}(1, -1)^\top$.

**Check.** $A \mathbf{v}_1 = \frac{1}{\sqrt 2}(3, 3)^\top = 3 \mathbf{v}_1$. $\checkmark$
$A \mathbf{v}_2 = \frac{1}{\sqrt 2}(1, -1)^\top = 1 \cdot \mathbf{v}_2$. $\checkmark$

**Geometric reading.** $A$ stretches by 3 along the diagonal direction
$(1,1)$ and leaves the anti-diagonal $(1,-1)$ unchanged. Since $A$ is
symmetric, those two directions are perpendicular (the spectral theorem
in action).

### Example 3 (harder) — SVD of a 2×2 by hand

Find the SVD of

$$
A = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}.
$$

**Step 1: compute $A^\top A$.**

$$
A^\top A = \begin{pmatrix} 3 & 4 \\ 0 & 5 \end{pmatrix} \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix} = \begin{pmatrix} 25 & 20 \\ 20 & 25 \end{pmatrix}.
$$

**Step 2: eigendecompose $A^\top A$.** Characteristic polynomial:
$(25-\lambda)^2 - 400 = 0$, so $25 - \lambda = \pm 20$, giving
$\lambda_1 = 45$, $\lambda_2 = 5$.

Singular values: $\sigma_1 = \sqrt{45} = 3\sqrt{5}$, $\sigma_2 = \sqrt{5}$.

Eigenvectors of $A^\top A$ (which become the right singular vectors, columns of $V$):

- For $\lambda_1 = 45$: $(A^\top A - 45 I)\mathbf{v} = \mathbf{0}$ gives
  $v_1 = v_2$, so $\mathbf{v}_1 = \frac{1}{\sqrt 2}(1, 1)^\top$.
- For $\lambda_2 = 5$: $v_1 = -v_2$, so $\mathbf{v}_2 = \frac{1}{\sqrt 2}(1, -1)^\top$.

**Step 3: compute $U$ via $\mathbf{u}_i = A \mathbf{v}_i / \sigma_i$.**

$$
A \mathbf{v}_1 = \frac{1}{\sqrt 2} A \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt 2} \begin{pmatrix} 3 \\ 9 \end{pmatrix}, \quad \mathbf{u}_1 = \frac{1}{3\sqrt 5} \cdot \frac{1}{\sqrt 2} \begin{pmatrix} 3 \\ 9 \end{pmatrix} = \frac{1}{\sqrt{10}} \begin{pmatrix} 1 \\ 3 \end{pmatrix}.
$$

$$
A \mathbf{v}_2 = \frac{1}{\sqrt 2} A \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt 2} \begin{pmatrix} 3 \\ -1 \end{pmatrix}, \quad \mathbf{u}_2 = \frac{1}{\sqrt 5} \cdot \frac{1}{\sqrt 2} \begin{pmatrix} 3 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{10}} \begin{pmatrix} 3 \\ -1 \end{pmatrix}.
$$

**Step 4: assemble.**

$$
U = \frac{1}{\sqrt{10}} \begin{pmatrix} 1 & 3 \\ 3 & -1 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 3\sqrt 5 & 0 \\ 0 & \sqrt 5 \end{pmatrix}, \quad V = \frac{1}{\sqrt 2} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}.
$$

You can verify $A = U\Sigma V^\top$ by direct multiplication; it works.

**What this tells you about $A$.** The matrix has condition number
$\sigma_1 / \sigma_2 = 3$. It stretches the $(1,1)/\sqrt 2$ direction by
$3\sqrt 5 \approx 6.7$ and the $(1,-1)/\sqrt 2$ direction by
$\sqrt 5 \approx 2.2$. The first stretch direction lands along $(1,3)/\sqrt{10}$
in the output; the second lands along $(3,-1)/\sqrt{10}$.

### Example 4 (runnable code) — power iteration for the dominant eigenvector

The classical algorithm for finding the largest-magnitude eigenvalue and
its eigenvector. The idea is dead simple: pick a random vector, multiply
by $A$ repeatedly, normalize each step. Whatever the random vector had
in the dominant-eigenvector direction grows like $\lambda_1^k$; everything
else grows like $\lambda_i^k$ with $|\lambda_i| < |\lambda_1|$, so it
gets dwarfed. After enough iterations, only the dominant direction is
visible.

```python
import numpy as np

def power_iteration(A: np.ndarray, num_iters: int = 1000, tol: float = 1e-10):
    """Estimate the dominant eigenvalue and eigenvector of a square matrix A.

    Assumes the largest-magnitude eigenvalue is unique and real.
    Returns (eigenvalue, eigenvector) where eigenvector has unit norm.
    """
    n = A.shape[0]
    # Step 1: start from a random unit vector. Any non-degenerate
    # starting vector works; randomness avoids accidentally picking
    # one orthogonal to the dominant eigenvector.
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    eigenvalue_prev = 0.0
    for k in range(num_iters):
        # Step 2: multiply by A. This pushes v toward the dominant
        # eigendirection because that direction is amplified the most.
        w = A @ v
        # Step 3: normalize so the vector doesn't blow up or vanish.
        v = w / np.linalg.norm(w)
        # Step 4: Rayleigh quotient — best estimate of eigenvalue given v.
        eigenvalue = v @ (A @ v)
        # Step 5: convergence check.
        if abs(eigenvalue - eigenvalue_prev) < tol:
            break
        eigenvalue_prev = eigenvalue

    return eigenvalue, v


if __name__ == "__main__":
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    lam, v = power_iteration(A)
    print(f"Estimated dominant eigenvalue: {lam:.6f}")
    print(f"Estimated dominant eigenvector: {v}")
    # Expected: lambda = 3.0, v = (1, 1)/sqrt(2) ≈ (0.7071, 0.7071)
    # (or its negation — eigenvectors are determined up to sign)

    # Verify against numpy:
    eigvals, eigvecs = np.linalg.eig(A)
    print(f"NumPy eigenvalues: {eigvals}")
    print(f"NumPy eigenvectors:\n{eigvecs}")
```

**Expected output:**

```
Estimated dominant eigenvalue: 3.000000
Estimated dominant eigenvector: [0.70710678 0.70710678]
NumPy eigenvalues: [3. 1.]
NumPy eigenvectors:
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]
```

**Line-by-line:** `A @ v` is matmul (3 multiplies and 2 adds — but
NumPy calls into BLAS even at this size). `np.linalg.norm` computes the
L2 norm. The **Rayleigh quotient** $\mathbf{v}^\top A \mathbf{v} / \mathbf{v}^\top \mathbf{v}$
gives the best eigenvalue estimate when $\mathbf{v}$ is the unit
eigenvector — it's a standard trick. Convergence rate is governed by
the ratio $|\lambda_2 / \lambda_1|$: in our example $1/3$, so we
converge fast.

**Failure modes:**

- If the starting vector is exactly orthogonal to the dominant
  eigenvector, we never converge to it (we converge to the second). In
  practice random starts make this measure-zero event vanish.
- If $|\lambda_1| = |\lambda_2|$ (e.g., $\lambda_1 = 1, \lambda_2 = -1$),
  the iterate oscillates.
- For *complex* dominant eigenvalues (real non-symmetric $A$), the
  iterate doesn't settle in $\mathbb{R}^n$.

Power iteration is what Google ran on the web graph for the first
PageRank implementations. The matrix had ~10 billion rows by then;
explicit eigendecomposition was impossible. Power iteration on a sparse
matrix-vector product was the only viable option.

---

## 8. Connections to Machine Learning, Deep Learning, and LLMs

Concrete mapping from today's objects to where they appear:

| Object | Where it appears |
|---|---|
| Vector | Token embedding (`d_model`-dim); image patch embedding; user/item vector in collaborative filtering. |
| Matrix as data | A mini-batch of $B$ examples of $d$ features is a $B \times d$ matrix. A weight matrix $W$ in `nn.Linear(d_in, d_out)` is shape $(d_{\text{out}}, d_{\text{in}})$. |
| Matrix as function | Every `nn.Linear` layer is exactly a matrix multiplication plus an additive bias. A network is a stack of these, separated by element-wise nonlinearities. |
| Matmul $AB$ | Forward pass of attention: $S = QK^\top / \sqrt{d_k}$ produces a $T \times T$ attention-score matrix from $T \times d_k$ queries and keys. |
| Transpose | The $K^\top$ in attention. The weight matrix transpose in backprop: $\nabla_{\mathbf{x}} = W^\top \nabla_{\mathbf{y}}$. The `.T` you see all over PyTorch code. |
| Inverse | Rarely used directly in deep learning (because we don't solve linear systems exactly during training). Common in classical ML: closed-form linear regression $\hat\beta = (X^\top X)^{-1} X^\top y$, Gaussian process covariance inversion. |
| Pseudoinverse | Behind `np.linalg.lstsq`. Used in classical regression when $X^\top X$ is singular or near-singular. Also used in some second-order optimization methods (natural gradient). |
| Determinant | Appears in normalizing flows (change-of-variables formula uses $\log |\det J|$, the Jacobian determinant), in Gaussian likelihoods, and in the log-volume terms of probabilistic models. |
| Trace | $\mathrm{tr}(A^\top B)$ is the Frobenius inner product; it shows up in regularizers ($\mathrm{tr}(W^\top W) = \|W\|_F^2$) and in many gradient identities. |
| Eigenvalues/eigenvectors | PageRank, spectral clustering, Hessian analysis of loss landscapes (positive eigenvalues = local minimum), neural network stability (largest eigenvalue of recurrent weight matrix governs whether activations explode or vanish). |
| SVD | **PCA** (Phase 3 / Day 6) is SVD of the centered data matrix. **LoRA** (Day 19) writes $\Delta W = BA$ with $A, B$ of low rank — a parameterized SVD truncation. **Model compression** factors layers via SVD and keeps top singular values. **Effective rank** of attention layers diagnoses whether heads have collapsed. |
| Rank | Determines expressiveness of low-rank approximations. The intrinsic-dimension paper (Aghajanyan et al., 2020) showed fine-tuning happens in a tiny subspace; LoRA exploits this. |
| Tensors (3D, 4D) | Every PyTorch activation. `(batch, seq, hidden)` for text. `(batch, channels, H, W)` for vision. `(batch, heads, seq, d_k)` for multi-head attention internals. |
| `einsum` | Universal notation for tensor contractions. Modern attention code (e.g., FlashAttention reference) is written almost entirely in `einsum`. |

**Worked connection: attention is matmul.**

```python
import torch
import torch.nn.functional as F

# Toy attention: batch=2, seq=4, d_model=8, n_heads=2 -> d_k=4
B, T, d_model, h = 2, 4, 8, 2
d_k = d_model // h

x = torch.randn(B, T, d_model)              # (B, T, d_model)
W_q = torch.randn(d_model, d_model)         # matmul
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

# Project. Each is a matmul over the last dim.
Q = x @ W_q  # (B, T, d_model)
K = x @ W_k
V = x @ W_v

# Split into heads via reshape -> transpose.
Q = Q.view(B, T, h, d_k).transpose(1, 2)    # (B, h, T, d_k)
K = K.view(B, T, h, d_k).transpose(1, 2)
V = V.view(B, T, h, d_k).transpose(1, 2)

# THE matmul: scores = Q K^T / sqrt(d_k).  (B, h, T, T)
scores = Q @ K.transpose(-1, -2) / (d_k ** 0.5)

# Softmax over the key axis.
attn = F.softmax(scores, dim=-1)            # (B, h, T, T)

# Weighted sum of values.
out = attn @ V                              # (B, h, T, d_k)

# Merge heads back.
out = out.transpose(1, 2).contiguous().view(B, T, d_model)  # (B, T, d_model)

print(out.shape)  # torch.Size([2, 4, 8])
```

Every operation in this block is something we covered today: matmul,
reshape (re-indexing a tensor), transpose (swapping two axes), one
softmax (the only nonlinearity), one matmul. **That is attention.** Day 13
will dissect this in detail; today, just notice that everything is matmul.

---

## 9. Practical Implementation (NumPy / PyTorch)

The day's API cheat sheet you should commit to muscle memory:

| Operation | NumPy | PyTorch |
|---|---|---|
| Create | `np.array([[1,2],[3,4]])` | `torch.tensor([[1.,2.],[3.,4.]])` |
| Shape | `A.shape` | `A.shape` (or `A.size()`) |
| Matmul | `A @ B` or `np.matmul(A, B)` | `A @ B` or `torch.matmul(A, B)` |
| Transpose | `A.T` (2D), `A.transpose(...)` (nD) | `A.T` (2D), `A.transpose(0, 1)`, `A.permute(...)` |
| Inverse | `np.linalg.inv(A)` | `torch.linalg.inv(A)` |
| Pseudoinverse | `np.linalg.pinv(A)` | `torch.linalg.pinv(A)` |
| Least squares | `np.linalg.lstsq(A, b, rcond=None)` | `torch.linalg.lstsq(A, b)` |
| Determinant | `np.linalg.det(A)` | `torch.linalg.det(A)` |
| Trace | `np.trace(A)` | `torch.trace(A)` |
| Eigendecomp | `np.linalg.eig(A)` (general), `np.linalg.eigh(A)` (symmetric, **prefer this**) | `torch.linalg.eig`, `torch.linalg.eigh` |
| SVD | `np.linalg.svd(A, full_matrices=False)` | `torch.linalg.svd(A, full_matrices=False)` |
| Norm | `np.linalg.norm(x)`, `np.linalg.norm(A, 'fro')` | `torch.linalg.norm(x)`, `torch.linalg.norm(A, 'fro')` |
| einsum | `np.einsum('ij,jk->ik', A, B)` | `torch.einsum('ij,jk->ik', A, B)` |
| Solve $Ax=b$ | `np.linalg.solve(A, b)` | `torch.linalg.solve(A, b)` |

**Numerical hygiene rules to adopt today:**

1. **Use `full_matrices=False` for SVD** unless you specifically need the
   full $m \times m$ and $n \times n$ orthogonal matrices. The "thin" SVD
   is much cheaper.
2. **Prefer `eigh` over `eig` when the matrix is symmetric.** `eigh`
   guarantees real outputs, uses a stable algorithm, and is faster.
3. **Never invert a matrix to solve $Ax=b$.** Use `solve` (LU with partial
   pivoting). Inverting and multiplying loses 1–2 digits of accuracy and
   is 2–3× slower.
4. **Use `lstsq` for over- or under-determined systems**, not pseudoinverse.
   `lstsq` is more numerically stable and faster.
5. **Add a small ridge to ill-conditioned matrices.** Instead of
   $(X^\top X)^{-1}$, compute $(X^\top X + \epsilon I)^{-1}$ with
   $\epsilon \sim 10^{-6}$. This is "ridge regularization" and saves a
   lot of grief.
6. **Print shapes.** When code breaks, the first debug step is to print
   `.shape` of every tensor involved.

### Today's lab exercises (do these now)

Each has a reference solution in a collapsed `<details>` block. Try first.
Aim 30–45 min per exercise.

**Exercise 1 — Matmul from scratch.**
Implement `my_matmul(A, B)` using only Python loops and NumPy element
indexing (no `@`, no `np.matmul`, no `np.dot`). Verify against `A @ B`
on random matrices.

<details>
<summary>Reference solution</summary>

```python
import numpy as np

def my_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    m, p = A.shape
    p2, n = B.shape
    assert p == p2, f"inner dims mismatch: {A.shape} @ {B.shape}"
    C = np.zeros((m, n), dtype=np.result_type(A, B))
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

# Test
rng = np.random.default_rng(0)
A = rng.standard_normal((5, 7))
B = rng.standard_normal((7, 3))
assert np.allclose(my_matmul(A, B), A @ B)
print("matmul passes")
```

Then time it against `A @ B` on a 200×200 matrix. You'll see ~1000× slowdown.
That gap is what BLAS buys you. *Reflection:* this is why every deep
learning framework calls BLAS. You will never write production matmuls;
you will use `@`. But knowing what `@` is doing under the hood is the
foundation of understanding everything else.

</details>

**Exercise 2 — Transpose from scratch, and verify $(AB)^\top = B^\top A^\top$.**
Implement `my_transpose(A)`. Then on random $A, B$ verify the identity
numerically with `np.allclose`.

<details>
<summary>Reference solution</summary>

```python
def my_transpose(A: np.ndarray) -> np.ndarray:
    m, n = A.shape
    AT = np.zeros((n, m), dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            AT[j, i] = A[i, j]
    return AT

A = rng.standard_normal((4, 6))
B = rng.standard_normal((6, 5))
lhs = my_transpose(A @ B)
rhs = my_transpose(B) @ my_transpose(A)
assert np.allclose(lhs, rhs)
print("transpose identity holds")
```

Note `np.transpose` and `A.T` don't actually move memory — they return a
view with swapped strides. That's why they're "free." Your for-loop
version actually shuffles bytes; production code never does.
</details>

**Exercise 3 — Pseudoinverse via SVD, and least-squares solution.**
Use `np.linalg.svd` to compute $A^+$ for a tall, thin matrix (e.g., 50×3).
Solve a noisy linear regression $A\mathbf{x} \approx \mathbf{b}$ both via
your pseudoinverse and via `np.linalg.lstsq`. Confirm both give the same
$\mathbf{x}$.

<details>
<summary>Reference solution</summary>

```python
def my_pinv(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Invert non-tiny singular values, zero out the rest for stability.
    s_inv = np.where(s > rcond * s.max(), 1.0 / s, 0.0)
    return Vt.T @ np.diag(s_inv) @ U.T

# Synthetic regression: 50 examples, 3 features, true beta = [1, -2, 3], noise.
rng = np.random.default_rng(0)
A = rng.standard_normal((50, 3))
beta_true = np.array([1.0, -2.0, 3.0])
b = A @ beta_true + 0.1 * rng.standard_normal(50)

beta_pinv = my_pinv(A) @ b
beta_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)

print("via my pinv :", beta_pinv)
print("via lstsq   :", beta_lstsq)
print("true beta   :", beta_true)
assert np.allclose(beta_pinv, beta_lstsq)
```

Expected: both estimates are within ~0.02 of `[1, -2, 3]`. The recovery
isn't exact because of noise; that's normal. Try increasing the sample
size from 50 to 5000 and watch the estimate sharpen.
</details>

**Exercise 4 — Power iteration on a 5×5 symmetric matrix.**
Construct a random symmetric 5×5 matrix (e.g., `M = (A + A.T)/2` for
random $A$). Run power iteration. Verify against `np.linalg.eigh`.

<details>
<summary>Reference solution</summary>

```python
def power_iteration(A, num_iters=1000, tol=1e-12):
    n = A.shape[0]
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    lam_prev = 0.0
    for _ in range(num_iters):
        w = A @ v
        v = w / np.linalg.norm(w)
        lam = v @ A @ v
        if abs(lam - lam_prev) < tol:
            break
        lam_prev = lam
    return lam, v

rng = np.random.default_rng(0)
M = rng.standard_normal((5, 5))
M = (M + M.T) / 2  # force symmetry
lam, v = power_iteration(M)
eigs, _ = np.linalg.eigh(M)
top_eig = eigs[np.argmax(np.abs(eigs))]
print(f"power iteration:   lambda = {lam:.6f}")
print(f"numpy ground truth: lambda = {top_eig:.6f}")
# Should match to 5–6 decimals.
```

If signs of eigenvectors disagree, that's fine — eigenvectors are
defined only up to scalar multiples.
</details>

**End-of-lab checkpoint:** all four files run, all assertions pass, and
you can describe in one sentence what each algorithm is doing.

---

## 10. Advanced and Research Frontier

(Skim this. You're not expected to deeply understand it on Day 1 — it's
here so you know where this material *goes*.)

- **Randomized SVD** (Halko, Martinsson, Tropp, 2011 — "Finding Structure
  with Randomness"). For huge matrices, exact SVD is infeasible. Random
  projections reduce the problem to a small one; the top singular values
  come out with provable accuracy. Used in every modern dimensionality-
  reduction library and inside LoRA's initialization.
- **Sketch-and-solve** least squares (Drineas–Mahoney). Approximate
  $\min \|Ax - b\|$ by sketching $A$ down with a random matrix; solve
  the small problem.
- **Tensor decompositions** (CP, Tucker, tensor train). Generalize SVD
  to rank-3+ tensors. Tensor-train decomposition powers tensor networks
  in quantum simulation and some efficient model architectures.
- **Communication-avoiding linear algebra.** On a GPU, the bottleneck
  is data movement, not arithmetic. Algorithms like the Cannon and
  SUMMA matmul minimize cross-device communication; FlashAttention
  (Dao et al., 2022) is a communication-avoiding *attention* algorithm
  that re-derives the same outputs with far less HBM traffic. We'll
  see this on Day 24.
- **Low-rank adaptation in LLMs.** Hu et al., 2021 — "LoRA: Low-Rank
  Adaptation of Large Language Models." Showed that fine-tuning updates
  $\Delta W$ have effectively low intrinsic rank, so you can train
  $r(m+n)$ parameters instead of $mn$ and reach equivalent performance.
  Direct application of the Eckart–Young theorem.
- **Spectral bias / NTK** (Jacot et al., 2018; Rahaman et al., 2019).
  The eigendecomposition of the **Neural Tangent Kernel** explains why
  neural networks fit low-frequency functions first.
- **The geometry of attention.** Recent (2024–2025) work analyzes
  attention as a kernel method and studies the spectrum of attention
  matrices to understand long-context generalization and to design
  better positional encodings (e.g., RoPE-extension methods).

---

## 11. Common Mistakes, Misconceptions, and Anti-Patterns

1. **Confusing $A \cdot B$ (element-wise) with $A B$ (matrix product).**
   In NumPy, `A * B` is Hadamard product (requires same shape). `A @ B`
   is matmul. In math notation, juxtaposition $AB$ means matrix product;
   $A \odot B$ is element-wise. Always double-check what your symbol
   means in context.
2. **Forgetting matmul is non-commutative.** $AB \ne BA$ in general.
   Many bugs are reversed factor orders.
3. **Treating a $1 \times n$ row vector and an $n$-element vector as
   interchangeable.** NumPy will sometimes broadcast them silently and
   produce a wrong-shape result. PyTorch is stricter but not bulletproof.
4. **Calling `inv` instead of `solve`.** Anywhere you'd write `inv(A) @ b`,
   write `solve(A, b)`. Faster and more stable.
5. **Computing the full SVD of a huge matrix.** $O(\min(m,n)^2 \max(m,n))$
   means SVD on a 50000×4096 matrix is *minutes*. Use `full_matrices=False`,
   and for top-$k$ use `scipy.sparse.linalg.svds` or randomized SVD.
6. **Assuming all eigenvalues are real.** True only for symmetric (or
   Hermitian) matrices. The covariance matrix of any dataset is
   symmetric PSD; this is the case that comes up 80% of the time in ML,
   which is why students develop the wrong intuition.
7. **Mis-reading shapes in attention code.** Modern transformer code has
   tensors of shape `(B, h, T, d_k)`. A wrong `transpose(1, 2)` vs
   `transpose(-1, -2)` is the most common bug. Always print shapes after
   each reshape.
8. **Treating "rank of a tensor" and "rank of a matrix" as the same.**
   For a 2D array, rank = dimension of the column space. For a $k$-D
   array, rank in NumPy means $k$ (number of dimensions). Two different
   concepts that share a word.
9. **Forgetting that $X^\top X$ is symmetric PSD.** Whenever you see this
   construction (in least squares, in Gram matrices, in covariance), you
   can use `eigh` and you know the eigenvalues are non-negative.
10. **Storing dense identity matrices for the sake of clarity.** Don't
    materialize `np.eye(n)` if you can avoid it; `+ epsilon` to the
    diagonal of an existing matrix is much cheaper.

---

## 12. Self-Assessment Questions

Answer each before opening the spoiler. Be honest with yourself — guessing
"yeah, kind of" is what destroys learning.

**Q1.** What is the difference between a vector and a 1×$n$ matrix?

<details><summary>Answer</summary>

Conceptually nothing — both store $n$ numbers. Computationally they're
distinct objects in most libraries: a 1D NumPy array has shape `(n,)`
while a row matrix has shape `(1, n)`. Broadcasting and matmul rules
differ slightly. Default to 1D arrays for "vectors of data" and 2D
arrays when shape matters.
</details>

**Q2.** Why is $(AB)^\top = B^\top A^\top$ instead of $A^\top B^\top$?

<details><summary>Answer</summary>

Because of dimensions. If $A$ is $m \times p$ and $B$ is $p \times n$,
then $AB$ is $m \times n$, so $(AB)^\top$ is $n \times m$. For the
right-hand side to also be $n \times m$, we need $B^\top$ ($n \times p$)
on the *left* of $A^\top$ ($p \times m$). The formula is forced by shape.
The full proof is §6.3.
</details>

**Q3.** Give an example of a 2×2 matrix that has no inverse.

<details><summary>Answer</summary>

$\begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$. Its determinant is $1 \cdot 4 - 2 \cdot 2 = 0$.
The second row is twice the first; the columns are linearly dependent;
the rank is 1, not 2. Geometrically, it maps the entire 2D plane onto a
single line through the origin, so the mapping isn't invertible.
</details>

**Q4.** Prove that if $A$ is symmetric, all its eigenvalues are real.
(Derivation question.)

<details><summary>Answer</summary>

Let $A \mathbf{v} = \lambda \mathbf{v}$ for a non-zero (possibly complex)
$\mathbf{v}$. Take the complex conjugate transpose (denoted $*$) of both
sides: $\mathbf{v}^* A^* = \bar\lambda \mathbf{v}^*$. Since $A$ is real
symmetric, $A^* = A^\top = A$, so $\mathbf{v}^* A = \bar\lambda \mathbf{v}^*$.
Multiply on the right by $\mathbf{v}$: $\mathbf{v}^* A \mathbf{v} = \bar\lambda \mathbf{v}^* \mathbf{v}$.

But also $\mathbf{v}^* A \mathbf{v} = \mathbf{v}^* (\lambda \mathbf{v}) = \lambda \mathbf{v}^* \mathbf{v}$.

So $\bar\lambda \mathbf{v}^* \mathbf{v} = \lambda \mathbf{v}^* \mathbf{v}$.

Since $\mathbf{v}^* \mathbf{v} = \|\mathbf{v}\|^2 > 0$, we can divide to
get $\bar\lambda = \lambda$, i.e., $\lambda$ is real. $\blacksquare$
</details>

**Q5.** What's the rank of $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{pmatrix}$?

<details><summary>Answer</summary>

Row 2 is twice row 1, so it's redundant. Row 1 and row 3 are linearly
independent (no scalar multiple connects them). So rank = 2.
</details>

**Q6.** For $A = \begin{pmatrix} 0 & 2 \\ 3 & 0 \end{pmatrix}$, what are
the singular values? (Derivation question.)

<details><summary>Answer</summary>

Compute $A^\top A = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$.
Eigenvalues are 9 and 4. Singular values are $\sqrt 9 = 3$ and $\sqrt 4 = 2$.
Note: even though $A$ has zeros on the diagonal, it's invertible — the
singular values are non-zero. The eigenvalues of $A$ itself are
$\pm \sqrt{-6} = \pm i\sqrt 6$, which are complex; this is why singular
values are the more robust descriptor.
</details>

**Q7.** A weight matrix $W$ has shape `(d_out, d_in) = (4096, 4096)`. You
want to replace it with a rank-16 approximation for LoRA. How many
parameters do you save?

<details><summary>Answer</summary>

Original: $4096 \times 4096 = 16{,}777{,}216$ parameters. Rank-16
approximation $W \approx BA$ with $A \in \mathbb{R}^{16 \times 4096}$
and $B \in \mathbb{R}^{4096 \times 16}$: $16 \cdot 4096 + 4096 \cdot 16 = 131{,}072$
parameters. Savings: ~128× fewer parameters. (LoRA actually freezes $W$
and learns only $BA$ as a *delta*, but the parameter count math is the same.)
</details>

**Q8.** What's the condition number of $A = \mathrm{diag}(1, 10^{-6})$,
and what does it mean operationally?

<details><summary>Answer</summary>

Singular values are 1 and $10^{-6}$, so $\kappa = 10^6$. Operationally:
solving $Ax = b$ in float32 (~7 digits) loses about 6 digits of accuracy
— you have about 1 digit of meaningful result. In float16 (~3 digits)
the answer is total noise. This is the practical reason mixed-precision
training (FP16) requires careful scaling.
</details>

**Q9.** (Implementation.) Write a function that computes the
Frobenius-norm best rank-$k$ approximation of a matrix $A$ using SVD.

<details><summary>Answer</summary>

```python
def low_rank_approx(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]
```

Verify on a random $10 \times 8$ matrix that
`np.linalg.matrix_rank(low_rank_approx(A, 3))` returns 3 and that the
Frobenius error `np.linalg.norm(A - low_rank_approx(A, k))` decreases
monotonically as $k$ increases. By Eckart–Young, this is the best rank-$k$
approximation in Frobenius norm; no other rank-$k$ matrix can do better.
</details>

**Q10.** (Implementation.) Using only `numpy` (no `np.linalg.inv`,
`np.linalg.solve`, `np.linalg.lstsq`), solve a small linear system
$Ax = b$ for a 3×3 invertible $A$ via your power-iteration / SVD code.

<details><summary>Answer</summary>

Trick: use SVD to build the pseudoinverse and apply it.

```python
def solve_via_svd(A, b):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt.T @ ((U.T @ b) / s)

A = np.array([[3., 2., -1.], [2., -2., 4.], [-1., .5, -1.]])
b = np.array([1., -2., 0.])
x = solve_via_svd(A, b)
print(x)
print(A @ x)  # should approx equal b
```

Mathematically: $A^{-1} = V \Sigma^{-1} U^\top$, and $\Sigma^{-1}$ on a
diagonal $\Sigma$ is just reciprocals. Note this uses `np.linalg.svd`
under the hood for SVD itself; the instruction was to avoid `inv`/`solve`/`lstsq`.
</details>

**Bonus — research level.** Why does FlashAttention need to recompute
attention scores on the backward pass instead of caching them like a
naive implementation? (You'll formally answer this on Day 24; today just
attempt an intuition.)

<details><summary>Hint</summary>

The cached attention-score matrix is $T \times T$, which for $T = 8192$
is 64 million entries per head per layer. For 32 layers × 32 heads,
caching all of them costs more memory than the model weights. FlashAttention
trades a small amount of recomputation for a large memory saving, and as
a side effect dramatically improves wall-clock time because the
recomputation happens entirely in fast SRAM rather than slow HBM.
</details>

---

## 13. Glossary

- **Basis** — a minimal set of linearly independent vectors whose
  combinations produce every vector in the space.
- **Column space** ($\mathcal{C}(A)$) — the span of the columns of $A$;
  the set of all possible outputs $A\mathbf{x}$.
- **Condition number** ($\kappa$) — ratio of largest to smallest
  singular value; measures sensitivity to numerical error.
- **Determinant** ($\det A$) — scalar; signed volume scaling factor of
  the transformation; zero iff $A$ is singular.
- **Diagonalizable** — a matrix is diagonalizable if it has a full set
  of linearly independent eigenvectors, allowing $A = PDP^{-1}$.
- **Eigenvalue** ($\lambda$) — scalar by which an eigenvector gets scaled
  under $A$.
- **Eigenvector** ($\mathbf{v}$) — non-zero vector satisfying $A\mathbf{v} = \lambda \mathbf{v}$.
- **einsum** — Einstein summation notation; concise way to write tensor
  contractions, e.g., `np.einsum('ij,jk->ik', A, B)` is matmul.
- **Frobenius norm** ($\|A\|_F$) — square root of the sum of squared
  entries; equals $\sqrt{\sum_i \sigma_i^2}$.
- **Hadamard product** ($A \odot B$) — element-wise multiplication.
- **Identity matrix** ($I_n$) — square matrix with 1s on diagonal, 0s
  elsewhere; the multiplicative identity for matmul.
- **Inverse** ($A^{-1}$) — for square non-singular $A$, the unique
  matrix with $AA^{-1} = I$.
- **Linear** — a function satisfying $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$.
- **Linearly independent** — no vector in a set is a linear combination
  of the others.
- **Matrix** — 2D array of numbers; equivalently, a linear function
  between vector spaces.
- **Norm** — non-negative scalar measure of vector "size." L2 norm:
  $\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}$.
- **Null space** ($\mathcal{N}(A)$) — set of vectors mapped to zero by $A$.
- **Orthogonal matrix** — square real matrix $Q$ with $Q^\top Q = I$;
  columns form an orthonormal basis; preserves lengths and angles.
- **Pseudoinverse** ($A^+$) — the Moore–Penrose generalized inverse;
  exists for every matrix; solves least-squares problems.
- **Rank** — dimension of the column space; number of non-zero singular
  values.
- **Rayleigh quotient** — $\mathbf{x}^\top A \mathbf{x} / \mathbf{x}^\top \mathbf{x}$;
  used as an eigenvalue estimator.
- **Scalar** — a single number.
- **Singular value** ($\sigma_i$) — non-negative entries on the diagonal
  of $\Sigma$ in the SVD; equal to $\sqrt{\lambda_i(A^\top A)}$.
- **Singular vector** — columns of $U$ (left) or $V$ (right) in $A = U\Sigma V^\top$.
- **Span** — the set of all linear combinations of a set of vectors.
- **Spectral theorem** — every real symmetric matrix has an orthonormal
  eigenbasis and real eigenvalues.
- **Spectrum** — the set of eigenvalues of a matrix.
- **SVD** (Singular Value Decomposition) — factorization $A = U\Sigma V^\top$;
  exists for every matrix.
- **Symmetric matrix** — square $A$ with $A^\top = A$.
- **Tensor** — multi-dimensional array; rank = number of indices needed
  to address one entry.
- **Trace** ($\mathrm{tr}\,A$) — sum of diagonal entries; equals sum of
  eigenvalues for square $A$.
- **Transpose** ($A^\top$) — matrix with rows and columns swapped.
- **Vector** — ordered list of numbers; element of $\mathbb{R}^n$.

---

## 14. Further Reading (specific pointers)

**Today, if you have extra time, watch one of these (don't try to read
the books cover-to-cover — that's months of work; today, just dip in).**

- **3Blue1Brown — "Essence of Linear Algebra" (YouTube series, 15
  videos, ~15 min each).** The single best visual introduction to
  linear algebra ever produced. For Day 1, the must-watch episodes are
  **Ch 1 (vectors)**, **Ch 3 (linear transformations and matrices)**,
  **Ch 4 (matrix multiplication as composition)**, **Ch 13–14
  (eigenvectors)**, and **Ch 16 (Abstract vector spaces — last one,
  optional)**. Total time: ~75 minutes for the must-watches.
- **Gilbert Strang — *Introduction to Linear Algebra*, 5th or 6th ed.**
  For today, **Chapter 1 (Introduction to Vectors)** and the first half of
  **Chapter 2 (Solving Linear Equations)**. Strang's MIT OCW course (MIT
  18.06, lecture videos on YouTube and OCW) tracks the book.
- **Trefethen & Bau — *Numerical Linear Algebra* (1997).** The
  numerical-analyst's perspective. **Lecture 4 (The Singular Value
  Decomposition)** and **Lecture 5 (More on the SVD)** are the cleanest
  SVD treatment in print. Read these *after* you've done today's lab.
- **Goodfellow, Bengio, Courville — *Deep Learning* (2016).** Free
  online at deeplearningbook.org. **Chapter 2 — Linear Algebra** is
  ML-focused and short (~30 pages); a fast supplementary read.
- **Karpathy — "micrograd" repository on GitHub** (
  `karpathy/micrograd`). 150-line scalar autograd engine. You won't use
  it today — you'll use it on Day 2. Skim the README so you know what's
  coming.

**Canonical papers (for context — don't try to read today).**

- Eckart, C. and Young, G. (1936) — *The approximation of one matrix by
  another of lower rank.* The theorem that justifies SVD truncation,
  PCA, and LoRA.
- Halko, Martinsson, Tropp (2011) — *Finding Structure with Randomness.*
  The randomized SVD reference.
- Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models.*
  The direct LoRA paper. Save this for Day 19.

---

## 15. The Next Topic

**Tomorrow (Day 2):** *Multivariable calculus, the chain rule, and
backpropagation as reverse-mode automatic differentiation.* You will
build a 30-line scalar autograd engine and train a 2-layer MLP on XOR.
Today's linear-algebra foundations are exactly what tomorrow's gradients
flow through.

**Why this order:** the chain rule for vector-valued functions is
expressed as Jacobian matrices, and backprop is the algorithm that
computes vector-Jacobian products in reverse. Without today's matmul
intuition, tomorrow's "$\nabla_x = W^\top \nabla_y$" looks magical;
with it, it's obvious.

**Branch alternatives if you want a different angle (don't take these
this week — finish the sprint first):**

- *Probability and information theory* (Day 3). You could swap Day 2 and
  Day 3 without loss, but doing calculus first makes Day 4's optimization
  cleaner.
- *Tensor decompositions (CP, Tucker, tensor train).* The generalization
  of SVD to higher ranks. Used in some efficient architectures but rare
  in mainline transformer work. Phase 8 territory.
- *Numerical linear algebra at depth (Krylov methods, Lanczos, Arnoldi).*
  Needed if you go into quantum chemistry / scientific computing. Not on
  the critical path to a multimodal LLM.

---

*End of Day 1. Commit your notes, commit your code, sleep well, and ask
your coach (me) any doubts. We pick up tomorrow at Day 2.*
