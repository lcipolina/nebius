# LLM Architectures Hometask 1 (PyTorch Optimization + BoW Baseline)

This folder contains an exported Python script version of a homework notebook:

- `llm_architectures_hometask_1.py` (main script)
- `LLM_Architectures,_hometask_1.ipynb` (original notebook export)
- `HOW_TO_RUN.md` (setup + run instructions)
- `REPORT.md` (writeup / conclusions)
- `plots/` (saved figures)

The headline topic in the script is:

> Optimization in PyTorch: Gradient Descent, SGD, numerical stability, and L1 regularization.

Below is a high-level description of what `llm_architectures_hometask_1.py` does.

## Leitmotif / End Goal

The recurring idea throughout the file is: **define a fixed feature schema once, then fill it per data point**.

In the Bag-of-Words (BoW) part, that means:

- Choose a vocabulary of size `V` from the training set. This fixes the **columns/features** (each vocab token gets a stable column index).
- For each document/sentence, create a length-`V` vector of counts (or 0/1 presence). This is **one row**.
- Stack all rows into a single design matrix:
  - `X ∈ R^{N×V}` where `N` = number of examples, `V` = vocabulary size

That matrix `X` is the end product that downstream models (here: logistic regression) train on. The optimization sections then focus on how different update rules (SGD, Momentum, AdaGrad, Adam) and choices (numerical stability, L1/L2 regularization) affect learning.

## What The Script Does (End To End)

### 1. Load A Binary Sentiment Dataset (SST-2)

The script uses Hugging Face `datasets`:

- Loads `SetFit/sst2` via `datasets.load_dataset`.
- Extracts:
  - `data_train = sst2["train"]`
  - `data_val = sst2["validation"]`

Each example includes:

- `text`: the raw sentence
- `label`: `0` (negative) or `1` (positive)


### 2. Clean Text For Consistent Tokenization

The script defines a `clean_text()` function and applies it to every example in both splits, creating a new column `clean_text`.

The cleaning steps:

- Lowercase all characters.
- Replace hyphens `-` with spaces.
- Remove most special characters while keeping letters, numbers, whitespace, and basic punctuation (`.,!?`).
- Collapse repeated whitespace and trim ends.

Why this exists:

- Bag-of-Words is very sensitive to surface form. Cleaning reduces “fake” vocabulary growth (for example, `Movie` vs `movie`).


### 3. Print Dataset Overview

The helper `print_dataset_overview()`:

- Prints train/validation sizes.
- Prints a few random cleaned examples and their labels.
- Prints label distribution (class balance) for both splits.

Why this exists:

- Class balance affects how you interpret metrics like accuracy and motivates metrics like F1 in imbalanced settings.


### 4. Build A Bag-of-Words (BoW) Feature Matrix

This section converts text into fixed-length numeric vectors suitable for classic ML models and simple neural baselines.

#### How Bag-of-Words Works (Complete Step-by-Step Example)

**INPUT: Raw Review**
```
"the movie is good it is"
```

---

**STEP 1: Tokenization** (split by whitespace)

Code:
```python
tokens = tokenize(review)  # review.split()
```

Output: A list of words
```
["the", "movie", "is", "good", "it", "is"]
```

Notice: "is" appears twice in the list. We're not removing duplicates—we need to count them later.

---

**STEP 2: Build Vocabulary from Entire Training Set**

First, we process **all** the training reviews and count word frequencies across the entire corpus:

Imagine training set:
```
Review 1: "the movie is good it is"
Review 2: "the movie is bad"
Review 3: "this movie is really good"
Review 4: "bad movie very bad"
... (thousands more reviews)
```

Count all words:
```python
token_counter = Counter()
for each_review in data_train:
    tokens = tokenize(each_review)
    token_counter.update(tokens)
```

Result (corpus-level frequencies):
```
"the":    50,000 times
"movie":  40,000 times
"is":     35,000 times
"good":   25,000 times
"bad":    20,000 times
"it":     15,000 times
"this":   10,000 times
"really": 8,500 times
"very":   7,200 times
"great":  6,500 times
```

**Keep only top 10 words** (the most frequent):
```python
vocab = {
    "the":    0,
    "movie":  1,
    "is":     2,
    "good":   3,
    "bad":    4,
    "it":     5,
    "this":   6,
    "really": 7,
    "very":   8,
    "great":  9
}
```

**Important:** The numbers (0-9) are **column indices**—they tell us where each word goes in the final vector.

---

**STEP 3: Vectorize Your Review Using This Vocabulary**

Now take your single review again:
```
review = "the movie is good it is"
tokens = ["the", "movie", "is", "good", "it", "is"]
```

Initialize an empty vector with 10 zeros (one position per vocab word):
```python
vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Loop through each token and **increment the count at the right position**:

```
Token 1: "the"   → vocab["the"] = 0      → vec[0] += 1  → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Token 2: "movie" → vocab["movie"] = 1    → vec[1] += 1  → [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
Token 3: "is"    → vocab["is"] = 2       → vec[2] += 1  → [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
Token 4: "good"  → vocab["good"] = 3     → vec[3] += 1  → [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
Token 5: "it"    → vocab["it"] = 5       → vec[5] += 1  → [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
Token 6: "is"    → vocab["is"] = 2       → vec[2] += 1  → [1, 1, 2, 1, 0, 1, 0, 0, 0, 0]
```

---

**STEP 4: Final Bag-of-Words Vector**

```python
vec = [1, 1, 2, 1, 0, 1, 0, 0, 0, 0]
```

**Interpretation:**
```
Position 0 (the):    count = 1
Position 1 (movie):  count = 1
Position 2 (is):     count = 2  ← appears twice!
Position 3 (good):   count = 1
Position 4 (bad):    count = 0  ← doesn't appear
Position 5 (it):     count = 1
Position 6 (this):   count = 0
Position 7 (really): count = 0
Position 8 (very):   count = 0
Position 9 (great):  count = 0
```

**This is your review as a machine-readable vector:** `[1, 1, 2, 1, 0, 1, 0, 0, 0, 0]`

Now a machine learning model can use this vector to make predictions!

---

**Summary of the Pipeline:**

1. **Tokenize** → convert text to list of words
2. **Count frequencies** (across all training data) → build vocabulary
3. **Vectorize each review** → tokenize it again, then count word occurrences using vocabulary indices
4. **Result** → fixed-length numeric vector for each review

---

Core components:

- `tokenize(text)`: whitespace split (a deliberately simple tokenizer).
- `build_vocabulary(data, top_k=10000)`:
  - Counts token frequencies across **training `clean_text` only** using `Counter`.
  - Keeps the `top_k` most frequent tokens.
  - Builds a mapping `vocab: token -> column_index`.
- `convert_text_to_vec(text, vocab)`:
  - Produces a vector of length `len(vocab)`.
  - Each index stores the count of the corresponding token in the sentence.
- `dataset_to_vec(data, vocab)`:
  - Applies the conversion to every row and stacks into a 2D matrix.

Outputs:

- `X_train`: shape `(n_train, vocab_size)` (count vectors)
- `y_train`: shape `(n_train,)` (float labels)
- `X_val`, `y_val` similarly

The script also prints:

- vocabulary size
- the matrix shape for training features
- one example vector and its non-zero entries (to show sparsity)

### 5. Part 1: Logistic Regression In PyTorch + Mini-batch SGD

This is the “Optimization in PyTorch” part of the assignment. The code is intentionally a *skeleton* with TODOs.

#### 5.1 Implement Logistic Regression (`nn.Module`)

The script defines a `LogisticRegression(nn.Module)` class skeleton that is intended to:

- Initialize parameters:
  - `w`: weights of shape `(n_features, 1)`
  - `b`: bias (scalar or shape `(1,)`)
  - Both should be `nn.Parameter` so PyTorch tracks gradients.
- `forward(x)`:
  - compute logits `x @ w + b`
  - apply `sigmoid(logits)` to get probabilities in `[0, 1]`
- `predict(x)`:
  - convert probabilities to 0/1 predictions using threshold `0.5`

**Logistic Regression: Inputs / Outputs / Goal / Method**

1) Inputs and outputs
- Input to the model (per example): a feature vector `x` of length `n_features` (here: a BoW row, so `n_features = vocab_size`).
- Model parameters (learned): weights `w` (one weight per feature) and bias `b`.
- Output (per example): a probability `p(y=1|x)` in `[0, 1]`, and optionally a hard class prediction by thresholding (e.g. `p >= 0.5`).

In matrix form for a batch/dataset:
- `X ∈ R^{N×V}` (N examples, V vocab features)
- `w ∈ R^{V×1}`, `b ∈ R`
- logits `z = Xw + b ∈ R^{N×1}`
- probabilities `p = sigmoid(z) ∈ [0,1]^{N×1}`

2) What logistic regression is trying to achieve
- Learn `w` and `b` so that the predicted probabilities match the true labels.
- In this task: estimate how likely a sentence is positive (label 1) vs negative (label 0) from BoW features.

3) Methodology
- Compute a linear score (logit): `z = x·w + b`.
- Convert the score to a probability with the sigmoid function: `p = 1 / (1 + exp(-z))`.
- Train by minimizing Binary Cross-Entropy (negative log-likelihood), typically using SGD/mini-batch SGD.
- Optionally add regularization:
  - L1 encourages sparsity in `w` (many weights driven toward 0).
  - L2 encourages small weights (smooth shrinkage).


#### 5.2 Binary Cross-Entropy Loss (Numerical Stability)

The script includes a placeholder `binary_cross_entropy_loss(y_pred, y_true)` for logistic regression.

This function is where “numerical stability” shows up:

- BCE uses `log(y_pred)` and `log(1 - y_pred)`.
- If `y_pred` becomes exactly `0` or `1`, logs become invalid (`log(0)`).
- Typical fix: clamp probabilities into `[epsilon, 1 - epsilon]` before taking logs.

#### 5.3 Train With Mini-batch SGD (Tracking History + Metrics)

The skeleton `sgd_logistic_regression(...)` is meant to:

1. Convert `X_train`, `y_train`, `X_val`, `y_val` to `torch.Tensor`.
2. Initialize the `LogisticRegression` model.
3. Create an SGD optimizer (`torch.optim.SGD`).
4. Train for multiple epochs:
   - Shuffle training data each epoch.
   - Loop over mini-batches.
   - Forward pass -> loss -> backward -> optimizer step.
   - Save the parameters (`w`, `b`) after each batch update into `history`.
5. Evaluate each epoch:
   - Compute train/val loss (non-regularized) and a metric (accuracy or F1).

Regularization hook (Task 1.5):

- The training loop includes a placeholder to add:
  - L1 penalty: `reg_lambda * ||w||_1`
  - L2 penalty: `reg_lambda * ||w||_2^2`

Important conceptual distinction used in the script:

- Training loss used for backprop may include regularization.
- Reported losses for comparison are often the *data loss* (non-regularized BCE) so runs are comparable.

### 6. Task 1.3: Hyperparameter Experiments (Placeholder)

The notebook assignment asks you to sweep:

- learning rates: `[0.01, 0.03, 0.1, 0.3, 1.0]`
- batch sizes: `[50, 100, 200]`

and visualize results with a heatmap (metric values by `(lr, batch_size)`).

In the `.py` export, this section is left as `# <Your code here>`.

### 7. Part 2: Compare Optimizers On Toy Functions

This section builds intuition for optimizer behavior by optimizing two 2D functions:

- A convex “bowl” function (easy, single global minimum).
- The six-hump camel function (non-convex, multiple local minima).

It provides skeleton implementations (TODOs) for:

- Gradient Descent
- Momentum
- AdaGrad
- Adam

For each optimizer, the expected outputs are:

- `trajectory`: the sequence of `(x_t, y_t)` parameter values over steps
- `values`: the objective function value over steps

There is also a plotting helper that draws contour plots and overlays optimizer trajectories for comparison.

### 8. Bonus: Intuition For L1 Regularization (And Proximal Descent)

The script ends with a markdown explanation covering:

- Why L1 adds a constant-magnitude push toward zero via `sign(w)`.
- Why that creates sparse solutions (many weights driven close to 0).
- Why plain gradient descent with L1 can “bounce” around 0.
- How proximal updates (soft-thresholding) can set small weights exactly to 0.
- Contrast with L2 regularization (smooth shrinkage proportional to the weight value).



---


## Appendix: PyTorch Tensors (What They Are And Why We Use Them)

### What Is A Tensor?

A **tensor** is PyTorch's main numeric container: an N-dimensional array.

- 0D tensor: scalar (single number)
- 1D tensor: vector
- 2D tensor: matrix
- higher-D tensors: used for batches, images, sequences, etc.

In this homework, you will mostly see:

- feature matrices like `X ∈ R^{N×V}` (2D tensors)
- parameter vectors/matrices like `w ∈ R^{V×1}` (2D tensor)
- labels like `y ∈ {0,1}^N` (often stored as float tensors for loss computation)

### Why Do We Need Tensors Here?

1. **CPU/GPU compatible computation**

PyTorch tensors can live on different devices (CPU or GPU) while using the same API. This is one reason PyTorch uses `torch.Tensor` instead of NumPy arrays for training loops.

2. **Autograd (automatic differentiation)**

PyTorch can automatically compute gradients through tensor operations. Conceptually:

- you compute predictions from tensors
- you compute a scalar loss
- `loss.backward()` uses **autograd** to compute gradients like `∂loss/∂w`

Important nuance:

- A tensor is just numbers by default.
- A tensor becomes a *learnable variable* when PyTorch is told to track gradients for it (commonly via `nn.Parameter` inside an `nn.Module`).

### Imports You Typically Need

```python
import torch
import torch.nn as nn
```

- `torch` provides the tensor type and math operations.
- `torch.nn` provides building blocks for models; `nn.Parameter` is the standard way to register a tensor as a trainable parameter.

### Initializing 1D And 2D Tensors (Shape + dtype)

In PyTorch you usually initialize tensors by specifying:

- `shape` (dimensions)
- `dtype` (number type, often `torch.float32` for learnable weights)

**1D (vector) examples**

```python
n = 5
v1 = torch.zeros(n, dtype=torch.float32)      # shape: (5,)
v2 = torch.randn(n, dtype=torch.float32)      # shape: (5,)
v3 = torch.tensor([1.0, 2.0, 3.0])            # shape: (3,)
```

**2D (matrix / column-vector) examples**

```python
n_features = 10
W1 = torch.zeros((n_features, 1), dtype=torch.float32)      # shape: (10, 1)
W2 = 0.01 * torch.randn((n_features, 1), dtype=torch.float32)  # small random init
M  = torch.zeros((3, 4), dtype=torch.float32)               # shape: (3, 4)
```

### Making A Tensor Trainable (Autograd + Optimizers)

Two common patterns:

1. **Inside an `nn.Module`** (most common): wrap the tensor as an `nn.Parameter` so it shows up in `model.parameters()` and receives gradients.
2. **Standalone tensor**: set `requires_grad=True` if you're doing manual optimization experiments.

Example idea (not specific to this homework's TODOs):

```python
W = torch.zeros((n_features, 1), dtype=torch.float32, requires_grad=True)
```

### Device (CPU vs GPU) Concept

Tensors have a device. You can move them between devices (when available):

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = W.to(device)
```

NumPy arrays do not have this device concept in NumPy itself (they live on CPU), which is why training code typically uses `torch.Tensor` rather than NumPy arrays.
