# -*- coding: utf-8 -*-
"""LLM Architectures, hometask 1.ipynb

Original file is located at
    https://colab.research.google.com/drive/1DVChXYQUPytStcp3Lmt8JGpZLwwXLaEz

# Optimization in PyTorch — Gradient Descent, SGD, Numerical Stability, and L1 Regularization
# This script builds a Bag-of-Words sentiment baseline (SST-2), then implements logistic regression + a mini-batch SGD training loop (with a numerically-stable BCE loss and optional L1/L2 penalties), and finally compares GD/Momentum/AdaGrad/Adam on convex vs non-convex toy functions.

STEPS PERFORMED IN THIS NOTEBOOK:
1- Download dataset and split in train/test
2- CLEAN text from symbols (train and test sets)
3- TOKENIZE -> each word is a token (train and test sets)
4- Count the number of occurences of each word (train and test sets)
5- BUILD VOCABULARY {token: index}:
Get the top-k by frequency to build our vocabulary (training set only)
This frequency is then thrown away after we have built our vocabulary
The vocabulary is a map from {token: index} that we will use to vectorize the data.
The index is just a column id in the final vector, it does not have any meaning by itself, it is just a way to assign a fixed position to each word in the vector.
6- VECTORIZE: each token in the vocabulary

HOW TO RUN THIS NOTEBOOK:
#!pip install datasets
# conda activate llm
# python3 llm_architectures_hometask_1.py

"""


import re
from collections import Counter
import random
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Plot output (always next to this script, regardless of current working directory)
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
SAVE_PLOTS = True
SHOW_PLOTS = False  # keep False to avoid interactive popups in scripts

# Heavy data pipeline (dataset download + vectorization). Turn off if you only
# want to run Part 2 (toy optimizer comparisons).
RUN_DATA_PIPELINE = True


def load_datasets(dataset_name="SetFit/sst2"):
    """Load sentiment analysis dataset from HuggingFace.

    Args:
        dataset_name (str): Name of the dataset to load from HuggingFace.
            Defaults to "SetFit/sst2" (Stanford Sentiment Treebank v2).

    Returns:
        tuple: A tuple containing (data_train, data_val) where each is a
            HuggingFace dataset object containing the training and validation
            splits respectively.
    """
    try:
        from datasets import load_dataset
        try:
            from datasets import DownloadMode  # type: ignore
        except Exception:  # pragma: no cover
            DownloadMode = None
    except Exception as e:  # pragma: no cover (environment-dependent)
        raise RuntimeError(
            "Failed to import HuggingFace `datasets`.\n"
            "This is almost always a version mismatch between `datasets`, `pyarrow`, and Python.\n"
            "Fix (pip):   `pip install -U datasets pyarrow`\n"
            "Fix (conda): `conda install -c conda-forge datasets pyarrow`\n"
            "If you only want to run Part 2 (toy optimizers), set `RUN_DATA_PIPELINE = False`."
        ) from e

    # HuggingFace `datasets` caches downloads by default. We also pin an explicit
    # cache directory and tell it to reuse the cached dataset if present.
    #
    # Tip: after one successful download, you can set `HF_DATASETS_OFFLINE=1`
    # to guarantee no network access is attempted.
    # Prefer the default HF cache (so we can reuse any existing downloads).
    # If you want a project-local cache, set `HF_DATASETS_CACHE_DIR` or create
    # the folder `./.hf_datasets_cache` next to this script.
    cache_dir_env = os.environ.get("HF_DATASETS_CACHE_DIR")
    local_cache_dir = Path(__file__).resolve().parent / ".hf_datasets_cache"

    load_kwargs = {}
    if cache_dir_env:
        load_kwargs["cache_dir"] = cache_dir_env
    elif local_cache_dir.exists():
        load_kwargs["cache_dir"] = str(local_cache_dir)
    if DownloadMode is not None:
        load_kwargs["download_mode"] = DownloadMode.REUSE_DATASET_IF_EXISTS
    sst2 = load_dataset(dataset_name, **load_kwargs)
    data_train = sst2["train"]
    data_val = sst2["validation"]
    return data_train, data_val

if RUN_DATA_PIPELINE:
    data_train, data_val = load_datasets()


"""**Text Cleaning**

Before converting text into a numerical representation (e.g., Bag-of-Words =  frequency count of words), it is important to apply text cleaning.
The goal of this step is to reduce noise and ensure that similar pieces of text are represented consistently.
We use the following cleaning function:
"""


def clean_text(text: str) -> str:
    """Clean and normalize text for natural language processing.

    Applies the following transformations:
    - Convert to lowercase
    - Replace hyphens with spaces
    - Remove special characters (keep only letters, numbers, spaces, and basic punctuation)
    - Collapse multiple spaces into single spaces
    - Strip leading and trailing whitespace

    Args:
        text (str): Raw input text to clean.

    Returns:
        str: Cleaned and normalized text.

    Example:
        >>> clean_text("Hello--World!!! @#$")
        'hello world!'
    """
    # Lowercase letters only
    text = text.lower()

    # Replace hyphens with space
    text = text.replace('-', ' ')

    # Keep letters, numbers, spaces, and basic punctuations
    # Remove everything else (like @ # $ % ^ & * ( ) etc.)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def _add_clean_text(example):
    return {"clean_text": clean_text(example["text"])}


X_train = None
y_train = None
X_val = None
y_val = None

if RUN_DATA_PIPELINE:
    # Apply cleaning (cached by `datasets` unless the function/code changes)
    data_train = data_train.map(_add_clean_text, load_from_cache_file=True)
    data_val = data_val.map(_add_clean_text, load_from_cache_file=True)

def print_dataset_overview(data_train, data_val, n_examples=5, seed=42):
    """Print summary statistics and sample examples from train/validation datasets.

    Displays:
    - Dataset sizes
    - Random examples from training set with their labels
    - Label distribution (counts and percentages) for both sets

    Args:
        data_train: Training dataset (HuggingFace dataset object).
        data_val: Validation dataset (HuggingFace dataset object).
        n_examples (int): Number of random examples to print. Defaults to 5.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        None. Prints to stdout.
    """
    rng = random.Random(seed)

    print(f"size of training set: {len(data_train)}")
    print(f"size of validation set: {len(data_val)}\n")

    for _ in range(n_examples):
        r = rng.randint(0, len(data_train) - 1)
        row = data_train[r]
        label_text = row.get("label_text", "positive" if row["label"] == 1 else "negative")
        print(f"{label_text} Text: {row['clean_text']}")

    train_counts = Counter(data_train["label"])
    val_counts = Counter(data_val["label"])

    train_total = len(data_train)
    val_total = len(data_val)

    print("\nTraining set label distribution:")
    print(f"Negative (0): {train_counts[0]} ({train_counts[0] / train_total:.2%})")
    print(f"Positive (1): {train_counts[1]} ({train_counts[1] / train_total:.2%})")

    print("\nValidation set label distribution:")
    print(f"Negative (0): {val_counts[0]} ({val_counts[0] / val_total:.2%})")
    print(f"Positive (1): {val_counts[1]} ({val_counts[1] / val_total:.2%})")


if RUN_DATA_PIPELINE:
    print_dataset_overview(data_train, data_val)

#===============================================================================
#===============================================================================

''' LCK comments:
Bag-of-Words pipeline summary (conceptual + implementation choices):
1) tokenize(text): sentence -> convert to list of words (we call them tokens).
2) build_vocabulary(data): count token frequencies with Counter (hash map),
  keep top_k words, map each word to a fixed index: word -> column id.

EXAMPLE of what we are doing here:
Cleaned sentence:
i liked this movie, this is good!

Tokens:
["i", "liked", "this", "movie,", "this", "is", "good!"]

Bag-of-words counts:
{"i": 1, "liked": 1, "this": 2, "movie,": 1, "is": 1, "good!": 1}

Vocabulary mapping (from training corpus, kept only top_k words):
FIRST - Each word is assigned a unique index (column id) in the feature vector:
{
    "good!": 0,
    "this": 1,
    "is": 2,
    "liked": 3,
    "i": 4,
    "movie,": 5
}

SECOND - COUNTS are placed in the vector according to the assigned indices:
Final Bag-of-Words vector:
[1, 2, 1, 1, 1, 1]  <- counts in order of vocabulary indices

Interpretation:
- Position 0 (good!):  count = 1
- Position 1 (this):   count = 2
- Position 2 (is):     count = 1
- Position 3 (liked):  count = 1
- Position 4 (i):      count = 1
- Position 5 (movie,): count = 1

Key insight: The BoW vector is ORDER-INVARIANT.
"i liked this movie, this is good!" → [1, 2, 1, 1, 1, 1]
"good! is this this, movie, liked i" → [1, 2, 1, 1, 1, 1]  (same vector, word order lost)
'''

#===============================================================================
#===============================================================================

"""We convert the text into numerical vectors that can be used as input for machine learning models.

1. Implementing the Bag-of-Words (BoW) representation building a vocabulary using only the training set.

2. Count the frequency of each token across the training corpus.

3. Keep only the top V=10,000 most frequent tokens (or fewer if memory is limited).

4. Convert each sentence into a sparse vector of token counts.

Each vector should represent how many times each vocabulary word appears in the sentence.

For example, if the vocabulary contains: ["movie", "good", "bad"]

Then the sentence: "good movie good"

Should become: [1,2,0]

**Note:** this is a simple bag of word implementation and not standart practice.

"""

def tokenize(text):
    """Split text into individual tokens (words) by whitespace.

    This is a simple baseline tokenizer that splits on whitespace and
    intentionally ignores word order. One whitespace-separated word
    is treated as one token.

    Args:
        text (str): Cleaned text to tokenize.

    Returns:
        list: List of string tokens.

    Example:
        >>> tokenize("this movie is good")
        ['this', 'movie', 'is', 'good']
    """
    return text.split()

'''BUIL THE VOCABULARY AND VECTORIZE THE DATA
1) build_vocabulary(data): count token frequencies with Counter (hash map),
   keep top_k words, map each word to a fixed index: word -> column id.

   Note that we don't build the hash map with every word in the English dictionary
   we only build it with the words that appear in our training set, and we only keep the most frequent ones to control feature size.

   In practice, we build a corpus-specific vocabulary from training data (often top-k), not from the whole dictionary.

   THIS IS SPECIFIC TO THE WAY WE ENGINEER THE DATA: In Bag-of-Words:

    Each vocabulary word is one feature.
    So if your vocabulary has 10,000 words, each sentence is represented by 10,000 features (a 10,000-dimensional vector).
    Each feature value is the count of that word in the sentence.

    NOTE ABOUT TERMINOLOGY:
    In classic ML (like your BoW + logistic regression), we explicitly design features:

    - one vocabulary word = one feature
    - feature value = word count

    In LLMs, we usually do not hand-engineer features like that.

    Instead:

    Input tokens are mapped to embeddings.
    Transformer layers automatically learn internal representations.
    Those internal activations are the model’s learned features (latent features).

    So:

    - BoW world: features are explicit, human-defined.
    - LLM world: features are implicit, learned by the network.
    People still use the word “feature” in LLMs, but it usually means learned representation dimensions/patterns, not manual columns like "word_x_count".

'''

# NEXT STEPS:
# - convert_text_to_vec(text, vocab): build a fixed-length count vector
#    (histogram) where each position stores the token frequency in the text.
#
# - dataset_to_vec(data, vocab): apply step (3) to all texts and stack rows
#    into a 2D feature matrix for model training.
#
# Important: this representation is order-invariant ("bag"), so it captures
# token presence/frequency but not word order or grammar.

def build_vocabulary(data, top_k=10000):
    """Build a fixed vocabulary from training data.

    Counts token frequencies across the corpus and keeps only the top_k
    most frequent tokens to control vocabulary size. Creates a mapping
    from token strings to fixed indices.

    Args:
        data: Dataset containing 'clean_text' field with tokenizable text.
        top_k (int): Maximum number of most frequent tokens to keep.
            Defaults to 10000.

    Returns:
        dict: Vocabulary mapping from token (str) to index (int).
    """
    token_counter = Counter()  # hash map to count token frequencies across the corpus
    for text in data['clean_text']:   # ← Go through each review (line by line)
        tokens = tokenize(text)       # ← Tokenize the review into a list of tokens (words)
        token_counter.update(tokens)  # ← Update the token frequency counts in the Counter with the tokens from this review
    most_common = token_counter.most_common(top_k) # Method from the Counter class to get the top_k most common tokens and their counts as a list of (token, count) tuples.
    vocab = {word: i for i, (word, _) in enumerate(most_common)}
    # Unpacks the tuples: (word, count) → uses word, ignores count (_)
    # Creates: {'good': 0, 'movie': 1, 'is': 2, ...}
    # We are getting the top words we need for our vocabulary.
    return vocab

def convert_text_to_vec(text, vocab):
    """Convert a single text to a Bag-of-Words count vector.

    Creates a fixed-length vector where each position corresponds to
    a vocabulary word and the value is the count of that word in the text.
    Words not in vocabulary are ignored.

    Args:
        text (str): Cleaned text to vectorize.
        vocab (dict): Vocabulary mapping from token (str) to index (int).

    Returns:
        numpy.ndarray: 1D array of shape (len(vocab),) with word counts.

    Example:
        >>> vocab = {'good': 0, 'movie': 1, 'bad': 2}
        >>> convert_text_to_vec('good movie good', vocab)
        array([2, 1, 0])
    """
    tokens = tokenize(text)
    vec = np.zeros(len(vocab), dtype=int)
    for token in tokens:
        if token in vocab:
            vec[vocab[token]] += 1
    return vec

def dataset_to_vec(data, vocab):
    """Convert all texts in a dataset to a 2D Bag-of-Words matrix.

    Applies convert_text_to_vec to each text in the dataset and stacks
    the resulting vectors into a 2D feature matrix for model training.

    Args:
        data: Dataset containing 'clean_text' field with texts to vectorize.
        vocab (dict): Vocabulary mapping from token (str) to index (int).

    Returns:
        numpy.ndarray: 2D array of shape (n_samples, vocab_size) where each
            row is the Bag-of-Words vector for one text.
    """
    vectors = []
    for text in data['clean_text']:
        vec = convert_text_to_vec(text, vocab)
        vectors.append(vec)
    return np.array(vectors)

def inspect_bow_vector(vectors, text, example_idx=0):
    """Inspect and visualize a single Bag-of-Words vector.

    Displays the full vector, non-zero entries with their indices and counts,
    original tokens, and compares token count to vector sparsity.

    Args:
        vectors (numpy.ndarray): 2D array of BoW vectors from dataset_to_vec.
        text (str): Cleaned text corresponding to the example vector.
        example_idx (int): Index of the example to inspect. Defaults to 0.

    Returns:
        None. Prints inspection results to stdout.
    """
    example = vectors[example_idx]
    print(f'\nExample vector: {example}')
    print("\nExample vector (non-zero entries):")

    indices = np.where(example > 0)
    for idx in indices[0]:
        print(f"index: {idx} | count: {example[idx]}")

    tokens = tokenize(text)
    print(f"\nExample tokens: {tokens}")
    print(f"Length of tokens: {len(tokens)}")
    print(f"Length of non-zero entries: {len(indices[0])}")


if RUN_DATA_PIPELINE:
    # Build vocabulary from training set
    vocab = build_vocabulary(data_train)
    print(f'Vocabulary size: {len(vocab)}')

    # Vectorize training data
    train_vectors = dataset_to_vec(data_train, vocab)
    print(f'Vectorized training data shape: {train_vectors.shape}')

    # Inspect first example
    inspect_bow_vector(train_vectors, data_train['clean_text'][0])

    """**Our training data for the task 1 will be:**"""

    X_train = train_vectors
    y_train = np.array(data_train["label"], dtype=np.float32)
    X_val = dataset_to_vec(data_val, vocab)
    y_val = np.array(data_val["label"], dtype=np.float32)

"""### **Part 1 - Implement SGD for Logistic Regression in PyTorch**

####**Task 1.1. - Implement Logistic Regression in PyTorch (2 points)**

In this task, you will implement a logistic regression model from scratch using PyTorch primitives.

The logistic regression prediction function is:

$\hat{Y} = \frac{1}{1+exp^{-(wx +b)}}$

Complete the class below.

You are required to implement:

1. weight initialization (with different options)
2. the forward pass (logits + sigmoid)
3. prediction logic (thresholding)
"""

'''
Each input example is a feature vector x with n_features entries (in your script: BoW counts, so one feature per vocab word).
Logistic regression assigns one weight per feature, so w must have n_features weights.
With batches, you want shapes that multiply cleanly:
x is typically (batch_size, n_features)
w is (n_features, 1)
so x @ w becomes (batch_size, 1) (one logit per example)

The weights are the result of training in the sense that they’re what you end up learning.

'''


class LogisticRegression(nn.Module):

    def __init__(self, n_features, init="zeros"):
        """
        Parameters
        ----------
        n_features : int
            Number of input features

        init : str or torch.Tensor
            Initialization method for weights:
            - "zeros"  -> initialize weights to zeros
            - "random" -> small random values (recommended scale ~0.01)
            - torch.Tensor -> use provided tensor
        """

        super().__init__()  # Parent class initialization (nn.Module)

        # Initialize the weight vector `w` based on the `init` argument
        # Make sure:
        # - shape is (n_features, 1)
        # - random initialization uses SMALL values (important!)
        # - if init is a tensor, clone + detach it

        if init == "zeros":
            w = torch.zeros((n_features, 1), dtype=torch.float32)      # shape: (n_features, 1)

        elif init == "random":
            w = torch.randn((n_features, 1), dtype=torch.float32) * 0.01  # shape: (n_features, 1)

        # If someone provides a tensor, we use it but we clone and detach to avoid potential issues with autograd if the original tensor requires gradients or is part of another computation graph.

        elif isinstance(init, torch.Tensor):
            w = init.clone().detach().to(dtype=torch.float32)  # ensure to convert whatever they provide to float32 for consistency

            # Ensure shape is (n_features, 1)
            if w.ndim == 1 and w.shape[0] == n_features:
                w = w.reshape(n_features, 1)
            elif w.ndim == 2 and w.shape == (n_features, 1):
                pass
            elif w.ndim == 2 and w.shape == (1, n_features):
                w = w.t()
            else:
                raise ValueError(
                    f"Provided init tensor must have shape (n_features, 1) or (n_features,), "
                    f"got {tuple(w.shape)} with n_features={n_features}"
                )

        else:
            raise ValueError("init must be 'zeros', 'random', or a torch.Tensor")

        # Wrap weights and bias using nn.Parameter
        # nn.Parameter is PyTorch’s way of marking a tensor as a learnable model parameter.
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))


    ''' Note about dimensions:
    Shape sanity check (typical case):

    x: (batch_size, n_features)
    w: (n_features, 1)
    b: (1,) (broadcasts)
    x @ w: (batch_size, 1)
    so logits: (batch_size, 1)

    '''

    def forward(self, x):
        """
        Forward pass

        Steps:
        1. Compute logits: x @ w + b
        2. Apply sigmoid to get probabilities

        Returns
        -------
        probs : torch.Tensor
            Values in range [0, 1]
        """

        logits = x @ self.w + self.b  # shape: (batch_size, 1)
        probs = torch.sigmoid(logits)  # shape: (batch_size, 1), values in [0, 1]

        return probs

    def predict(self, x):
        """
        Convert probabilities to class predictions

        Rule:
        - class 1 if p >= 0.5
        - class 0 otherwise
        """

        probs = self.forward(x)
        preds = (probs >= 0.5).float()  # shape: (batch_size, 1), values in {0.0, 1.0}

        return preds



"""####**Task 1.2 - Train Logistic Regression with SGD Using Your Previous Implementations (1 point)**

In this task, you will train the logistic regression model you implemented earlier using mini-batch stochastic gradient descent (SGD).

You must use your LogisticRegression class from Task 1.1.

The goal of this task is to practice building a full training loop in PyTorch while keeping the model and loss implementations modular.

Your function should
1. Initialize a LogisticRegression model
2. Train it on the training set using mini-batch SGD
3. Record the training log-loss after each epoch
4. Compute and report evaluation metrics on both the training and validation sets after each epoch.
You may choose any evaluation metric you find appropriate, such as accuracy, precision, recall, or F1-score, but you must briefly explain why this metric is suitable for this task.
5. Save the model parameters w and b after each batch update into a history log
6. Return:
   * the final trained parameters w and b
   * the batch-wise history of w and b


Bellow is a suggested skeleton you may revised
"""

def binary_cross_entropy_loss(y_pred, y_true):
    """Compute numerically stable binary cross-entropy loss.

    Computes BCE loss: -[y*log(p) + (1-y)*log(1-p)] where p is the
    predicted probability. Clamping is applied for numerical stability
    to avoid log(0).

    Args:
        y_pred (torch.Tensor): Predicted probabilities in range [0, 1].
            Shape: (n_samples, 1) or (n_samples,).
        y_true (torch.Tensor): Binary ground truth labels {0, 1}.
            Shape: (n_samples, 1) or (n_samples,).

    Returns:
        torch.Tensor: Scalar loss value (mean BCE across batch).
    """

    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)  # clamp forces values to be within [epsilon, 1-epsilon] to avoid log(0) which would be -inf and cause NaNs in gradients.

    # Return a scalar so `.backward()` works in the training loop.
    # We need to use 'torch.log' to ensure we get a tensor output that supports autograd, and we compute the mean loss across the batch for stability.
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()

'''
Epochs = how many full passes through the entire training set you do.

epochs=20 means: the model sees every training example 20 times (in some order).
One pass usually isn't enough for the model to learn good parameters, so we repeat the process multiple times.


Within each epoch, you iterate over the dataset in mini-batches.

Each batch is a chunk of the training set (size batch_size, except maybe the last one).
For each batch you do: forward → loss → backward → update.
So the structure is:

for epoch in epochs:
    shuffle training data
    for each batch:
    update weights using only that batch
    That’s “mini-batch SGD.”


'''
def sgd_logistic_regression(
    X_train, y_train,
    X_val, y_val,
    lr=0.01,
    epochs=20,
    batch_size=100,
    init="zeros",
    penalty='none',
    reg_lambda=0.0,
    metric='accuracy',
    print_metrics=False,
    log_history=True,
    history_weights_idx=None,
    history_stride=1,
):
    """Train logistic regression using mini-batch stochastic gradient descent.

    Trains a logistic regression model via SGD with optional L1/L2 regularization.
    Tracks training history at batch and epoch levels, with periodic validation
    and metric computation.

    Args:
        X_train (numpy.ndarray): Training features of shape (n_train, n_features).
        y_train (numpy.ndarray): Training labels of shape (n_train,) with values in {0, 1}.
        X_val (numpy.ndarray): Validation features of shape (n_val, n_features).
        y_val (numpy.ndarray): Validation labels of shape (n_val,) with values in {0, 1}.
        lr (float): Learning rate for SGD. Defaults to 0.01.
        epochs (int): Number of epochs (full passes over training data).
            Defaults to 20.
        batch_size (int): Mini-batch size. Defaults to 100.
        init (str): Weight initialization method ('zeros' or 'random').
            Defaults to 'zeros'.
        penalty (str): Regularization type ('none', 'l1', or 'l2').
            Defaults to 'none'.
        reg_lambda (float): Regularization coefficient. Defaults to 0.0.
        metric (str): Evaluation metric ('accuracy' or 'f1'). Defaults to 'accuracy'.
        print_metrics (bool): If True, print metrics after each epoch.
            Defaults to False.
        log_history (bool): If True, store parameter snapshots after batch updates.
            Defaults to True.
        history_weights_idx (sequence[int] | None): If provided, store only the
            selected weight indices in the history (saves memory). Defaults to None.
        history_stride (int): Store history every N batch updates (1 = every update).
            Defaults to 1.

    Returns:
        tuple: (w, b, history, epoch_log) where:
            - w (numpy.ndarray): Final learned weights of shape (n_features, 1).
            - b (numpy.ndarray): Final learned bias (scalar).
            - history (list): Batch-wise history with keys ['epoch', 'batch_start', 'w', 'b'].
            - epoch_log (list): Per-epoch logs with keys ['epoch', 'train_loss',
                'val_loss', 'train_metric', 'val_metric'].
    """


    # 1. Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    n_samples, n_features = X_train_tensor.shape

    # 2. Initialize model
    model = LogisticRegression(n_features=n_features, init=init)

    # 3. Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 4. Create logs
    history = []      # save w, b after batch updates
    epoch_log = []    # save epoch-level loss and metrics
    global_step = 0


    # 5. Training loop
    for epoch in range(epochs):

        # Shuffle the training data at the beginning of each epoch
        # This is important for SGD to ensure that batches are different each epoch and to help with convergence.
        # The way we do this is by generating a random permutation of the indices of the training samples and then reordering both X_train_tensor and y_train_tensor according to this permutation.
        # Note that Row i in X must always stay paired with row i in y.

        perm = torch.randperm(n_samples)  # generates a random permutation of indices from 0 to n_samples-1
        X_train_epoch = X_train_tensor[perm]  # shuffle the training features according to the random permutation
        y_train_epoch = y_train_tensor[perm]  # shuffle the training labels in the same way to maintain correct feature-label pairs

        # Iterate over mini-batches
        for start in range(0, n_samples, batch_size):

            end = start + batch_size

            # Select mini-batch
            X_batch = X_train_epoch[start:end]
            y_batch = y_train_epoch[start:end]

            # Forward pass
            y_pred = model(X_batch)

            # Compute non-regularized BCE loss using your function
            data_loss = binary_cross_entropy_loss(y_pred, y_batch)

            # Add regularization if needed
            # penalty == 'l1'  -> reg_lambda * ||w||_1
            # penalty == 'l2'  -> reg_lambda * ||w||_2^2
            # penalty == 'none' -> no regularization
            reg_term = data_loss.new_tensor(0.0)
            if penalty == 'l1':
                reg_term = reg_lambda * model.w.abs().sum()
            elif penalty == 'l2':
                reg_term = reg_lambda * (model.w ** 2).sum()
            elif penalty != 'none':
                raise ValueError("penalty must be one of: 'none', 'l1', 'l2'")

            loss = data_loss + reg_term

            # Backward pass and optimization step
            loss.backward()  # compute gradients of loss w.r.t. model parameters (w and b)
            optimizer.step()  # update model parameters using the computed gradients and the learning rate
            optimizer.zero_grad()  # reset gradients to zero for the next iteration (important to prevent gradient accumulation)

            # Save current parameter values after the batch update
            if log_history and (global_step % history_stride == 0):
                if history_weights_idx is None:
                    w_snapshot = model.w.detach().clone()
                else:
                    w_snapshot = model.w.detach()[history_weights_idx].clone().reshape(-1)

                history.append({
                    'step': global_step,
                    'epoch': epoch,
                    'batch_start': start,
                    'w': w_snapshot,
                    'b': model.b.detach().clone(),
                    'w_idx': None if history_weights_idx is None else list(history_weights_idx),
                })

            global_step += 1


        # 6. Epoch-level evaluation
        with torch.no_grad():

            # Compute probabilities on full train/val sets
            y_pred_train = model(X_train_tensor)
            y_pred_val = model(X_val_tensor)

            # Compute NON-regularized train loss and val loss
            train_loss = binary_cross_entropy_loss(y_pred_train, y_train_tensor)
            val_loss = binary_cross_entropy_loss(y_pred_val, y_val_tensor)

            # Convert probabilities to binary predictions
            y_hat_train = model.predict(X_train_tensor)
            y_hat_val = model.predict(X_val_tensor)


            # We can't overwrite y_train_tensor and y_val_tensor because they are used for the loss computation, which expects them to be of shape (n_samples, 1). If we flatten them to (n_samples,) for metrics, it would cause issues when we try to compute the loss in the next epoch. So we should create new variables for the flattened versions used in metrics.
            # We use native PyTorch operations to compute metrics to avoid unnecessary conversions to numpy and back, which can be inefficient.
            # Use flatten/reshape (not view) to be robust to non-contiguous tensors.
            y_train_flat = y_train_tensor.reshape(-1).float()  # shape: (n_samples,)
            y_val_flat = y_val_tensor.reshape(-1).float()      # shape: (n_samples,)
            y_hat_train_flat = y_hat_train.reshape(-1).float()  # shape: (n_samples,)
            y_hat_val_flat = y_hat_val.reshape(-1).float()      # shape: (n_samples,)

            # Compute evaluation metrics
            if metric == 'f1':
                # F1 calculation in native PyTorch
                tp_train = (y_hat_train_flat * y_train_flat).sum()
                precision_train = tp_train / (y_hat_train_flat.sum() + 1e-15)
                recall_train = tp_train / (y_train_flat.sum() + 1e-15)
                f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train + 1e-15)

                tp_val = (y_hat_val_flat * y_val_flat).sum()
                precision_val = tp_val / (y_hat_val_flat.sum() + 1e-15)
                recall_val = tp_val / (y_val_flat.sum() + 1e-15)
                f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-15)

                train_metric = f1_train.item()
                val_metric = f1_val.item()
            else: #accuracy calculation in pure torch  (avoid converting to numpy for metrics if we can do it in torch)
                train_metric = (y_train_flat == y_hat_train_flat).float().mean().item()
                val_metric = (y_val_flat == y_hat_val_flat).float().mean().item()

        epoch_log.append({
            'epoch': epoch,
            'train_loss': train_loss.item(),
            'val_loss': val_loss.item(),
            'train_metric': train_metric,
            'val_metric': val_metric
        })

        if print_metrics:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"Train {metric}: {train_metric:.4f} | "
                f"Val {metric}: {val_metric:.4f}"
            )

    return model.w.detach().numpy(), model.b.detach().numpy(), history, epoch_log



"""####**Task 1.3 - Experiments (2 points)**

Run multiple experiments with different combinations of:
* learning rates - [0.01, 0.03, 0.1 , 0.3 , 1.0]
* batch sizes - [50, 100 , 200]

For each experiment, record the final train and validation evaluation metric and the log-loss.

**Visualization:**

Present your results using a heatmap, where:
* X-axis: learning rate
* Y-axis: batch size
* Values: evaluation metric (train / validation)


**Analysis:**

Explain (in text) how learning rate and batch size affect:
- convergence speed
- stability of training
- final performance

Support your explanation using the patterns observed in your heatmap.
"""

# -----------------------
# Plotting helpers
# -----------------------

def _slugify(text: str) -> str:
    text = text.strip().lower()
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    out = []
    for ch in text.replace("—", "-").replace("–", "-").replace(" ", "_"):
        out.append(ch if ch in allowed else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "plot"


def _finalize_figure(fig, filename: Optional[str]):
    if SAVE_PLOTS and filename:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_loss_curves(epoch_log, title, filename):
    """Plot train/val BCE loss vs epoch from `sgd_logistic_regression` logs."""
    epochs = [e["epoch"] for e in epoch_log]
    train_loss = [e["train_loss"] for e in epoch_log]
    val_loss = [e["val_loss"] for e in epoch_log]

    plt.figure(figsize=(7, 4))
    fig = plt.gcf()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _finalize_figure(fig, filename)


def _plot_heatmap(values, x_labels, y_labels, title, cmap="viridis", filename=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(values, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels([str(x) for x in x_labels])
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([str(y) for y in y_labels])

    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Batch size")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if filename is None:
        filename = f"{_slugify(title)}.png"
    _finalize_figure(fig, filename)


def run_task_1_3_experiments(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rates=(0.01, 0.03, 0.1, 0.3, 1.0),
    batch_sizes=(50, 100, 200),
    epochs=20,
    init="zeros",
    metric="accuracy",
    capture_example=None,
):
    """Grid-search learning rate and batch size; return heatmap-ready arrays.

    If `capture_example=(batch_size, lr)` is provided, also return the per-epoch
    `epoch_log` for that configuration (useful to plot loss-vs-epoch curves).
    """
    train_metric = np.zeros((len(batch_sizes), len(learning_rates)), dtype=np.float32)
    val_metric = np.zeros((len(batch_sizes), len(learning_rates)), dtype=np.float32)
    train_loss = np.zeros((len(batch_sizes), len(learning_rates)), dtype=np.float32)
    val_loss = np.zeros((len(batch_sizes), len(learning_rates)), dtype=np.float32)
    example_epoch_log = None

    for i, batch_size in enumerate(batch_sizes):
        for j, lr in enumerate(learning_rates):
            _, _, _, epoch_log = sgd_logistic_regression(
                X_train,
                y_train,
                X_val,
                y_val,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                init=init,
                penalty="none",
                reg_lambda=0.0,
                metric=metric,
                print_metrics=False,
                log_history=False,
            )

            final = epoch_log[-1]
            train_metric[i, j] = final["train_metric"]
            val_metric[i, j] = final["val_metric"]
            train_loss[i, j] = final["train_loss"]
            val_loss[i, j] = final["val_loss"]

            if capture_example is not None and example_epoch_log is None:
                ex_bs, ex_lr = capture_example
                if batch_size == ex_bs and float(lr) == float(ex_lr):
                    example_epoch_log = epoch_log

    return {
        "learning_rates": list(learning_rates),
        "batch_sizes": list(batch_sizes),
        "train_metric": train_metric,
        "val_metric": val_metric,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "example_epoch_log": example_epoch_log,
    }


RUN_TASK_1_3 = True  # set True to run the sweep + plots
if RUN_DATA_PIPELINE and RUN_TASK_1_3:
    results_1_3 = run_task_1_3_experiments(
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rates=(0.01, 0.03, 0.1, 0.3, 1.0),
        batch_sizes=(50, 100, 200),
        epochs=20,
        init="zeros",
        metric="accuracy",
        capture_example=(100, 0.3),
    )

    _plot_heatmap(
        results_1_3["train_metric"],
        results_1_3["learning_rates"],
        results_1_3["batch_sizes"],
        title="Task 1.3 — Train metric",
        cmap="Blues",
    )
    _plot_heatmap(
        results_1_3["val_metric"],
        results_1_3["learning_rates"],
        results_1_3["batch_sizes"],
        title="Task 1.3 — Validation metric",
        cmap="Greens",
    )
    _plot_heatmap(
        results_1_3["train_loss"],
        results_1_3["learning_rates"],
        results_1_3["batch_sizes"],
        title="Task 1.3 — Train log-loss (BCE)",
        cmap="Reds",
    )
    _plot_heatmap(
        results_1_3["val_loss"],
        results_1_3["learning_rates"],
        results_1_3["batch_sizes"],
        title="Task 1.3 — Validation log-loss (BCE)",
        cmap="Oranges",
    )

    if results_1_3.get("example_epoch_log") is not None:
        plot_loss_curves(
            results_1_3["example_epoch_log"],
            title="Task 1.3 — Loss vs epoch (example: lr=0.3, batch=100)",
            filename="task_1_3_loss_vs_epoch_example_lr_0_3_bs_100.png",
        )

"""####**Task 1.4 — L1 Regularization and Sparsity (2 points)**

In this task, you will extend your implementation from Task 1.3 (SGD training) to include **L1 regularization**, and study how it affects the model.

**What is Regularization/ Penalty**

When there are too many features, some features might not be so important at all, but if we keep it, and try to fit our model to it perfectly, then it might overfit, trying to capture noisy (irrelevant) data or patterns. To reduce this overfitting so the model generalizes well and remove noisy data we use **regularization**. In linear models, mostly these regularization techniques are used:

* **L1 Penalty**: adding $\lambda *\sum{|w_i|}$ to the loss function

* **L2 Penalty**: adding $\lambda *\sum{||w||_i^2}$ to the loss function


**Why is this important?**

L1 *sparsifies* data, with part of weights being pushed strongly towards zero (with the right optimization technique, these weights become almost zero).
This leads to implicit feature selection.

In contrast, L2 shrinks weights but rarely makes them exactly zero.

**Task**

1. Modify your `sgd_logistic_regression` function from Task 1 to include L1 penalty.

2. Compare weight initialization:

   Try initializing the weight vector w in two different ways:
   * All zeros
   * Small random values

   Compare:
    * Stability (does training diverge? NaNs?)
    * Final performance
    * Sparsity (how many weights go to zero, use a small tolerance like 1e-7). Note that you'll unlikely get zeros. You'd need special optimization methods such as *proximal descent* to get true feature elimination; with SGD you'll still make part of the weights really small, so you'll still observe the pattern.

3. Study the effect of $\lambda$

   Run experiments with:

          reg_lambda = [0,1e-4,1e-3,1e-2,1e-1]

     Keep other parameters fixed (recommended):

      *  lr = 0.1
      *  batch_size = 100

     For each λ, record:
      * Train metric (for example accuracy or F1)
      * Validation metric
      * Number of non-zero weights that exceed a small threshold such as 1e-7

4. Visualization

     Plot:

     * number of non-zero weights vs lambda
     * train metric vs lambda
     * for a subset of features that get eliminated by the l1 regularization, training dynamics of their weights (weight vs step)

5. Write a small paragraph summarizing your insights
"""

def _count_nonzero_weights(w, tol=1e-4):
    w = np.asarray(w).reshape(-1)
    return int((np.abs(w) > tol).sum())


def run_task_1_4_l1_sparsity(
    X_train,
    y_train,
    X_val,
    y_val,
    reg_lambdas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
    lr=0.1,
    batch_size=100,
    epochs=20,
    init_options=("zeros", "random"),
    metric="accuracy",
    tol=1e-4,
):
    """Run L1-regularized SGD for multiple lambdas; return per-init results."""
    results = {}
    for init in init_options:
        init_records = []
        for lam in reg_lambdas:
            w, b, _, epoch_log = sgd_logistic_regression(
                X_train,
                y_train,
                X_val,
                y_val,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                init=init,
                penalty="l1",
                reg_lambda=float(lam),
                metric=metric,
                print_metrics=False,
                log_history=False,
            )

            final = epoch_log[-1]
            init_records.append(
                {
                    "reg_lambda": float(lam),
                    "train_metric": float(final["train_metric"]),
                    "val_metric": float(final["val_metric"]),
                    "train_loss": float(final["train_loss"]),
                    "val_loss": float(final["val_loss"]),
                    "nonzero_weights": _count_nonzero_weights(w, tol=tol),
                    "w": w,
                    "b": b,
                }
            )

        results[init] = init_records

    return results


def _plot_task_1_4_curves(task_1_4_results, metric_name="accuracy"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for init, records in task_1_4_results.items():
        lambdas = [r["reg_lambda"] for r in records]
        train_metric = [r["train_metric"] for r in records]
        val_metric = [r["val_metric"] for r in records]
        nonzero = [r["nonzero_weights"] for r in records]

        axes[0].plot(lambdas, nonzero, marker="o", label=f"init={init}")
        axes[1].plot(lambdas, train_metric, marker="o", label=f"init={init}")
        axes[2].plot(lambdas, val_metric, marker="o", label=f"init={init}")

    for ax in axes:
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlabel("reg_lambda (L1)")
        ax.grid(True)
        ax.legend()

    axes[0].set_title("Non-zero weights vs lambda")
    axes[0].set_ylabel(f"count(|w| > tol)")

    axes[1].set_title(f"Train {metric_name} vs lambda")
    axes[1].set_ylabel(metric_name)

    axes[2].set_title(f"Val {metric_name} vs lambda")
    axes[2].set_ylabel(metric_name)

    fig.tight_layout()
    _finalize_figure(fig, f"task_1_4_l1_summary_{_slugify(metric_name)}.png")


def run_task_1_4_weight_dynamics(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    lr=0.1,
    batch_size=100,
    epochs=20,
    init="random",
    metric="accuracy",
    reg_lambda=1e-1,
    tol=1e-4,
    n_weights_to_track=5,
    seed=42,
):
    """Track a few weights that end up near zero under L1 regularization."""
    rng = np.random.default_rng(seed)

    w, _, _, _ = sgd_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        init=init,
        penalty="l1",
        reg_lambda=float(reg_lambda),
        metric=metric,
        print_metrics=False,
        log_history=False,
    )

    w_flat = np.asarray(w).reshape(-1)
    # With plain SGD + L1, weights rarely become *exactly* 0. To make the
    # "eliminated features" plot reliable, track the smallest-|w| weights,
    # preferring those below `tol` if they exist.
    eliminated = np.where(np.abs(w_flat) <= tol)[0]
    if eliminated.size >= 1:
        candidates = eliminated
    else:
        candidates = np.argsort(np.abs(w_flat))[: max(n_weights_to_track, 50)]

    tracked = rng.choice(candidates, size=min(n_weights_to_track, candidates.size), replace=False)

    _, _, history, _ = sgd_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        init=init,
        penalty="l1",
        reg_lambda=float(reg_lambda),
        metric=metric,
        print_metrics=False,
        log_history=True,
        history_weights_idx=tracked.tolist(),
    )

    steps = [h["step"] for h in history]
    W = torch.stack([h["w"].detach().cpu().reshape(-1) for h in history]).numpy()

    plt.figure(figsize=(10, 4))
    fig = plt.gcf()
    for col, idx in enumerate(tracked):
        plt.plot(steps, W[:, col], label=f"w[{int(idx)}]")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("batch update step")
    plt.ylabel("weight value")
    plt.title(f"Task 1.4 — L1 weight dynamics (lambda={reg_lambda}, init={init})")
    plt.grid(True)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    _finalize_figure(fig, f"task_1_4_l1_weight_dynamics_lambda_{_slugify(str(reg_lambda))}_init_{_slugify(init)}.png")


RUN_TASK_1_4 = True  # set True to run the L1 experiments + plots
if RUN_DATA_PIPELINE and RUN_TASK_1_4:
    results_1_4 = run_task_1_4_l1_sparsity(
        X_train,
        y_train,
        X_val,
        y_val,
        reg_lambdas=(0.0, 1e-4, 1e-3, 1e-2, 1e-1),
        lr=0.1,
        batch_size=100,
        epochs=20,
        init_options=("zeros", "random"),
        metric="accuracy",
        tol=1e-4,
    )
    _plot_task_1_4_curves(results_1_4, metric_name="accuracy")

    run_task_1_4_weight_dynamics(
        X_train,
        y_train,
        X_val,
        y_val,
        lr=0.1,
        batch_size=100,
        epochs=20,
        init="random",
        metric="accuracy",
        reg_lambda=1e-1,
        tol=1e-4,
        n_weights_to_track=5,
    )

"""### **Part 2 - Comparing Optimization Algorithms on a Simple vs. Difficult Function (3 points)**

In this task, you will implement and compare several optimization algorithms on two different mathematical functions.

An optimization algorithm is a method used to update the model's parameters (weights and bias) in order to minimize the loss function.

At each step, it uses the gradients (how the loss changes) to decide:
- in which direction to move
- and how big the update should be

Different optimizers (like SGD, Adam) differ in how they use the gradients and how they control the step size, which affects how fast and how stably the model learns.

The purpose of this task is to help you build intuition for how optimization behaves in:

1. a simple convex function

2. a function with a narrow curved valley

You will implement the following optimizers:

* Gradient Descent (GD)
* Momentum
* AdaGrad
* Adam

Use your optimizers on the following functions:

Function A — Convex bowl:

$f(x,y)= x^2 +4y^2$

This is a simple convex function with a single global minimum.

Function B — the Six-hump Camel function:

$$
\left(4-2.1x^2+\frac{x^4}{3}\right)x^2+xy+\left(-4+4y^2\right)y^2.
$$

This function is much harder to optimize. It has two global minima: $(0.0898, -0.7126)$ and $(-0.0898,0.7126)$, with value about $-1.0316$ - and also several local minina.

Start from at least one non-optimal initial point, for example: (-2, -1.5)

**Task**

For each optimizer and for each function:

1. Initialize the parameters (x,y)
2. repeatedly compute the gradient
3. Update the parameters according to the optimizer rule
4. Record:
  * the function value at each iteration
  * the parameter values $(x_t,y_t)$ at each iteration

**Goal**

Compare how the different optimizers behave on:

* a simple convex function
* a difficult non-convex function

In particular, observe:

* how quickly each optimizer converges
* whether the optimization path is smooth or oscillatory
* whether an optimizer that works well on the convex bowl also works well on Camel
* how the geometry of the function affects the optimization process

**Coding**

Fill in the gaps in the implementation of the optimization method. By the way, you might find this code very repetitive. Can you find a way of shrinking it and getting rid of repetitions?

**Visualization**

Create the following plots:
1. Function value vs. iteration number
2. Optimization trajectories in the (x,y) plane. Use the `plot_trajectories_camel_log` function or create your own function to make an animation/gif.

For each question, please plot the behaviour of all optimization methods on one plot so that you could compare them.

**Analysis**

Summarize your findings in 2-3 text paragraphs. Things to ponder:

* Which optimizer performs best on the convex bowl?
* Which optimizer performs best on the Camel function? Will any of them reliably find global optima or will all of them get trapped in local minina from time to time, depending on the starting point.
* Do the same hyperparameters work equally well for both functions?
* What advantages do Momentum, AdaGrad, and Adam provide compared to plain Gradient Descent?
"""



def bowl(theta):
    """Simple convex quadratic function (bowl-shaped).

    Objective function: f(x,y) = x^2 + 4*y^2

    This is a simple convex function with a single global minimum at (0, 0).
    Used as a baseline for testing optimization algorithms.

    Args:
        theta (torch.Tensor): 2D parameter vector of shape (*, 2).

    Returns:
        torch.Tensor: Function values of shape (*,).
    """
    x, y = theta[..., 0], theta[..., 1]
    return x**2 + 4 * y**2


def camel(theta):
    """Six-hump Camel function - non-convex optimization benchmark.

    Objective function: f(x,y) = (4 - 2.1*x^2 + x^4/3)*x^2 + xy + (-4 + 4*y^2)*y^2

    This challenging non-convex function has two global minima at approximately
    (±0.0898, ∓0.7126) with value ≈ -1.0316, plus several local minima.
    Used to test optimizer robustness on complex landscapes.

    Args:
        theta (torch.Tensor): 2D parameter vector of shape (*, 2).

    Returns:
        torch.Tensor: Function values of shape (*,).
    """
    x, y = theta[..., 0], theta[..., 1]
    return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2


def plot_trajectories_camel_log(f, results, xlim=(-3, 3), ylim=(-2, 2),
                                  title="Optimization Trajectories", filename=None):
    """Plot optimization trajectories overlaid on log-scale function contours.

    Visualizes how different optimizers traverse the parameter space by
    plotting their trajectories on top of log-scaled contour lines of the
    objective function.

    Args:
        f (callable): Objective function that accepts torch.Tensor of shape (*, 2).
        results (dict): Dictionary mapping optimizer names to tuples of
            (trajectory, values) where trajectory is shape (n_steps, 2).
        xlim (tuple): X-axis limits (min, max). Defaults to (-3, 3).
        ylim (tuple): Y-axis limits (min, max). Defaults to (-2, 2).
        title (str): Title for the plot. Defaults to "Optimization Trajectories".

    Returns:
        None. Displays plot via matplotlib.
    """
    x_values = np.linspace(xlim[0], xlim[1], 400)
    y_values = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x_values, y_values)

    grid = np.stack((X, Y), axis=-1)
    Z = f(torch.tensor(grid, dtype=torch.float32)).detach().numpy()

    plt.figure(figsize=(8, 6))
    fig = plt.gcf()
    # Camel has negative values; shift before taking log to keep contours real-valued.
    Z_shift = Z - Z.min()
    plt.contour(X, Y, np.log1p(Z_shift + 1e-12), levels=30)

    for name, (trajectory, _) in results.items():
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', markersize=2, label=name)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if filename is None:
        filename = f"{_slugify(title)}.png"
    _finalize_figure(fig, filename)

def gradient_descent(f, theta0, lr=0.001, n_steps=2000):
    """Optimize using vanilla gradient descent.

    Updates parameters as: θ_t+1 = θ_t - lr * ∇f(θ_t)

    Args:
        f (callable): Objective function to minimize.
        theta0 (array-like): Initial parameters of shape (2,).
        lr (float): Learning rate. Defaults to 0.001.
        n_steps (int): Number of optimization steps. Defaults to 2000.

    Returns:
        tuple: (trajectory, values) where:
            - trajectory (torch.Tensor): Parameter values at each step, shape (n_steps+1, 2).
            - values (list): Function values at each step.
    """
    theta = torch.as_tensor(theta0, dtype=torch.float32).clone().detach().requires_grad_(True)

    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            theta -= lr * theta.grad

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(f(theta).item())

    return torch.stack(trajectory), values

def momentum(f, theta0, lr=0.001, beta=0.9, n_steps=2000):
    """Optimize using gradient descent with momentum.

    Maintains a velocity vector that accumulates gradients:
    - v_t+1 = β*v_t + ∇f(θ_t)
    - θ_t+1 = θ_t - lr*v_t+1

    This helps accelerate convergence and dampen oscillations.

    Args:
        f (callable): Objective function to minimize.
        theta0 (array-like): Initial parameters of shape (2,).
        lr (float): Learning rate. Defaults to 0.001.
        beta (float): Momentum coefficient (0 < beta < 1). Defaults to 0.9.
        n_steps (int): Number of optimization steps. Defaults to 2000.

    Returns:
        tuple: (trajectory, values) where:
            - trajectory (torch.Tensor): Parameter values at each step, shape (n_steps+1, 2).
            - values (list): Function values at each step.
    """
    theta = torch.as_tensor(theta0, dtype=torch.float32).clone().detach().requires_grad_(True)
    v = torch.zeros_like(theta)

    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            v = beta * v + theta.grad
            theta -= lr * v

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(f(theta).item())

    return torch.stack(trajectory), values

def adagrad(f, theta0, lr=0.1, eps=1e-8, n_steps=2000):
    """Optimize using AdaGrad (adaptive gradient descent).

    Adapts learning rate per parameter based on accumulated squared gradients:
    - G_t+1 = G_t + (∇f(θ_t))^2
    - θ_t+1 = θ_t - (lr / sqrt(G_t+1 + eps)) * ∇f(θ_t)

    Parameters with frequent gradients get smaller updates.

    Args:
        f (callable): Objective function to minimize.
        theta0 (array-like): Initial parameters of shape (2,).
        lr (float): Base learning rate. Defaults to 0.1.
        eps (float): Epsilon for numerical stability. Defaults to 1e-8.
        n_steps (int): Number of optimization steps. Defaults to 2000.

    Returns:
        tuple: (trajectory, values) where:
            - trajectory (torch.Tensor): Parameter values at each step, shape (n_steps+1, 2).
            - values (list): Function values at each step.
    """
    theta = torch.as_tensor(theta0, dtype=torch.float32).clone().detach().requires_grad_(True)
    G = torch.zeros_like(theta)

    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(n_steps):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            G = G + theta.grad ** 2
            theta -= (lr / (torch.sqrt(G) + eps)) * theta.grad

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(f(theta).item())

    return torch.stack(trajectory), values

def adam(f, theta0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=2000):
    """Optimize using Adam (Adaptive Moment Estimation).

    Combines momentum and adaptive learning rates using first and second moment estimates:
    - m_t+1 = β1*m_t + (1-β1)*∇f(θ_t)
    - v_t+1 = β2*v_t + (1-β2)*(∇f(θ_t))^2
    - m̂_t+1 = m_t+1 / (1 - β1^t)
    - v̂_t+1 = v_t+1 / (1 - β2^t)
    - θ_t+1 = θ_t - (lr / sqrt(v̂_t+1 + eps)) * m̂_t+1

    Args:
        f (callable): Objective function to minimize.
        theta0 (array-like): Initial parameters of shape (2,).
        lr (float): Learning rate. Defaults to 0.01.
        beta1 (float): Exponential decay rate for 1st moment (0 < beta1 < 1).
            Defaults to 0.9.
        beta2 (float): Exponential decay rate for 2nd moment (0 < beta2 < 1).
            Defaults to 0.999.
        eps (float): Epsilon for numerical stability. Defaults to 1e-8.
        n_steps (int): Number of optimization steps. Defaults to 2000.

    Returns:
        tuple: (trajectory, values) where:
            - trajectory (torch.Tensor): Parameter values at each step, shape (n_steps+1, 2).
            - values (list): Function values at each step.
    """
    theta = torch.as_tensor(theta0, dtype=torch.float32).clone().detach().requires_grad_(True)
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)

    trajectory = [theta.detach().clone()]
    values = [f(theta).item()]

    for step in range(1, n_steps + 1):
        loss = f(theta)
        loss.backward()

        with torch.no_grad():
            g = theta.grad
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)

            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)

            theta -= lr * m_hat / (torch.sqrt(v_hat) + eps)

        theta.grad.zero_()

        trajectory.append(theta.detach().clone())
        values.append(f(theta).item())

    return torch.stack(trajectory), values

def _plot_optimizer_values(results, title):
    plt.figure(figsize=(8, 4))
    fig = plt.gcf()
    for name, (_, values) in results.items():
        plt.plot(values, label=name)
    plt.xlabel("iteration")
    plt.ylabel("f(theta)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _finalize_figure(fig, f"{_slugify(title)}.png")


RUN_PART_2 = True  # set True to run optimizer comparisons + plots
if RUN_PART_2:
    # Start from a non-optimal point (per the task statement)
    theta0 = (-2.0, -1.5)

    results_bowl = {
        "GD": gradient_descent(bowl, theta0, lr=0.1, n_steps=2000),
        "Momentum": momentum(bowl, theta0, lr=0.1, beta=0.9, n_steps=2000),
        "AdaGrad": adagrad(bowl, theta0, lr=0.4, n_steps=2000),
        "Adam": adam(bowl, theta0, lr=0.1, n_steps=2000),
    }

    results_camel = {
        "GD": gradient_descent(camel, theta0, lr=0.005, n_steps=8000),
        "Momentum": momentum(camel, theta0, lr=0.005, beta=0.9, n_steps=8000),
        "AdaGrad": adagrad(camel, theta0, lr=0.05, n_steps=8000),
        "Adam": adam(camel, theta0, lr=0.02, n_steps=8000),
    }

    _plot_optimizer_values(results_bowl, title="Part 2 — Bowl: function value vs iteration")
    plot_trajectories_camel_log(bowl, results_bowl, xlim=(-3, 3), ylim=(-3, 3), title="Part 2 — Bowl trajectories")

    _plot_optimizer_values(results_camel, title="Part 2 — Camel: function value vs iteration")
    plot_trajectories_camel_log(camel, results_camel, xlim=(-3, 3), ylim=(-2, 2), title="Part 2 — Camel trajectories")



"""# Bonus. An attempt at explaining the L1 regularization phenomenon + a bit about proximal descent

The antigradient of the regularized loss $\mathcal{L}_{reg}(w) = \mathcal{L}(w) + \lambda\|w\|_1$ is
$$-\nabla_w\mathcal{L}_{reg}(w) = -\nabla_w\mathcal{L}(w) - \lambda\cdot \mathrm{sign}(w),$$
where $\mathrm{sgn}$ is the elementwise sign. This means that during the gradient descent the $i$-th coordinate of $w$ changes as
$$w_i \mapsto w_i - \alpha\frac{\partial}{\partial w_i}\mathcal{L}(w) + \begin{cases}
-\alpha\lambda,\mbox{ if $w_i > 0$},\\
+\alpha\lambda,\mbox{ if $w_i < 0$},
\end{cases},$$
where $\alpha$ is the step size. In other words, the rightmost summand pushes our $w_i$ towards $0$ with force $\alpha\lambda$. Now, imagine that the $i$-th feature is not very important. In this case, most likely, $\frac{\partial}{\partial w_i}\mathcal{L}(w)$ is small (change in $w_i$ doesn't change the loss much). So, the dominant force is the $\pm\alpha\lambda$, which may explain the almost-linear trajectories.

Of course, since $\alpha\lambda$ is constant and doesn't depend on $w_i$. This prevents us from converging to zero and explains the final noisy behaviour of $w_i$. We just leap around the origin.

I'd like to add here that if we used proximal descent, the step would become a two-step procedure like this:

$$w_i \mapsto w_i - \alpha\frac{\partial}{\partial w_i}\mathcal{L}(w),\\
w_i\mapsto\begin{cases}
w_i - \alpha\lambda,\mbox{ if $w_i \geqslant \alpha\lambda$},\\
0,\mbox{ if $|w_i| > \alpha\lambda$},\\
w_i + \alpha\lambda,\mbox{ if $w_i \leqslant -\alpha\lambda$},\\
\end{cases}$$
This way, small values of $w_i$ will be automatically zeroed, and they will only be able to escape zero again if the gradient push them hard enough.

**Comparison with L2 regularization**. For L2 regularization, the gradient step for the $i$-th coordinate would be
$$w_i \mapsto w_i - \alpha\frac{\partial}{\partial w_i}\mathcal{L}(w) - 2\alpha\lambda w_i$$
The rightmost term here depends on $w_i$; the closer it is to zero, the less influential it is, and the more important $\alpha\frac{\partial}{\partial w_i}\mathcal{L}(w)$ becomes. However, if the $i$-th feature is so worthless that $\alpha\frac{\partial}{\partial w_i}\mathcal{L}(w)\approx 0$, the process
$$w_i \mapsto w_i -  2\alpha\lambda w_i$$
converges to zero in contrast with the leaping behaviour of
$$w_i\mapsto w_i \pm\alpha\lambda$$
"""

# Script end marker (helps when running long sweeps).
print("All enabled tasks finished. Plots saved to plots/")
