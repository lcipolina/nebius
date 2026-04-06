# Report — Hometask 1 (Optimization in PyTorch)

## 0. Overview

This homework builds a simple sentiment classifier using a **Bag-of-Words (BoW)** representation and **logistic regression** trained with mini-batch SGD, then studies:

1. How **learning rate** and **batch size** affect convergence and final performance (Task 1.3).
2. How **L1 regularization** impacts sparsity and accuracy (Task 1.4).
3. How different **optimization algorithms** behave on an easy convex function vs a hard non-convex one (Part 2).

All plots referenced below are saved under `plots/`.

---

## 1. Setup and data pipeline

### 1.1 Dataset

We use SST-2 (`SetFit/sst2`) from HuggingFace `datasets`, with:

- Train: 6920 examples
- Validation: 872 examples

The dataset is approximately balanced in both splits (~50/50 positive/negative), so **accuracy** is a reasonable primary metric (F1 is also fine but less necessary under balance).

### 1.2 Preprocessing and features (BoW)

Steps:

1. Clean text (lowercase, remove special symbols, normalize whitespace).
2. Tokenize by whitespace.
3. Build a vocabulary from train only (top 10,000 tokens by frequency).
4. Vectorize each sentence into a length-10,000 count vector.

This produces:

- `X_train ∈ R^{6920×10000}`
- `X_val ∈ R^{872×10000}`

BoW is intentionally simple: it ignores word order and captures only token counts/presence.

---

## 2. Part 1 — Logistic regression + SGD

### 2.1 Model and loss

Model: logistic regression

- logits: `x @ w + b`
- probability: `sigmoid(logits)`
- prediction: threshold at 0.5

Loss: numerically stable binary cross-entropy (BCE) with clamping to avoid `log(0)` issues.

### 2.2 Training loop

We train with mini-batch SGD:

- shuffle each epoch
- forward → BCE loss (+ optional regularization) → backward → update
- report **non-regularized** train/val BCE for comparability across runs

---

## 3. Task 1.3 — Learning rate × batch size sweep

### 3.1 What was run

Grid:

- learning rates: `[0.01, 0.03, 0.1, 0.3, 1.0]`
- batch sizes: `[50, 100, 200]`
- epochs: `20`
- metric: `accuracy`

Plots:

- Train metric heatmap: `plots/task_1_3_-_train_metric.png`
- Validation metric heatmap: `plots/task_1_3_-_validation_metric.png`
- Train BCE heatmap: `plots/task_1_3_-_train_log-loss_bce.png`
- Validation BCE heatmap: `plots/task_1_3_-_validation_log-loss_bce.png`
- Example loss curve (lr=0.3, batch=100): `plots/task_1_3_loss_vs_epoch_example_lr_0_3_bs_100.png`

### 3.2 Observations (from the heatmaps)

**Learning rate dominates**:

- At low learning rates (`0.01`, `0.03`), both accuracy and loss are noticeably worse after 20 epochs → these configurations are under-trained within the fixed epoch budget.
- Increasing `lr` generally improves training accuracy and reduces BCE.

**Batch size interacts with learning rate**:

- Smaller batches (e.g., 50) tend to reach higher *training* accuracy at high `lr`, but can also encourage overfitting (train improves more than validation).
- Larger batches (e.g., 200) can be slightly more stable but may converge slower for the same number of epochs.

### 3.3 “Did we train enough?” (loss vs epoch)

The loss curve for an example good configuration (lr=0.3, batch=100) shows:

- Train loss steadily decreases across epochs.
- Validation loss drops quickly early, then flattens/plateaus and fluctuates slightly in later epochs.

See: `plots/task_1_3_loss_vs_epoch_example_lr_0_3_bs_100.png`

Interpretation:

- For `lr≈0.3`, 20 epochs is *likely sufficient* to reach a plateau on validation loss.
- For small learning rates, 20 epochs is not enough to reach comparable performance (their heatmap cells look systematically worse). If the goal is to compare learning rates fairly, increase epochs (e.g., 50–100) or use early stopping based on validation loss.

---

## 4. Task 1.4 — L1 regularization and sparsity

### 4.1 What was run

Parameters:

- L1 penalty
- `reg_lambda ∈ [0, 1e-4, 1e-3, 1e-2, 1e-1]`
- `lr = 0.1`, `batch_size = 100`, `epochs = 20`
- init: `zeros` vs `random`
- sparsity threshold (effective zero): `tol = 1e-4`

Plots:

- Summary (non-zero weights + train/val accuracy vs lambda):
  - `plots/task_1_4_l1_summary_accuracy.png`
- Weight dynamics for small-|w| features:
  - `plots/task_1_4_l1_weight_dynamics_lambda_0_1_init_random.png`

### 4.2 Observations

**Accuracy vs λ**:

- As `reg_lambda` increases, both train and validation accuracy drop.
- This is expected: stronger L1 penalizes weights more aggressively, which reduces model capacity.

**Sparsity in practice (SGD + L1)**:

- With plain SGD + L1, weights rarely become *exactly* zero; instead, many can become *small*.
- The “non-zero weights vs lambda” curve depends strongly on the chosen tolerance `tol` used to define “effectively zero”.
  - Using a very small tolerance (like `1e-7`) can misleadingly show “no sparsity” even if weights shrink.
  - Using `tol=1e-4` is more realistic for SGD noise at this scale.

**Weight dynamics**:

- The weight-dynamics plot tracks weights with the smallest magnitudes. This helps visualize the “push toward zero” effect of L1 even when weights are not strictly zero.

Takeaway: L1 provides a clear shrinkage effect and can support implicit feature selection, but (without proximal methods) sparsity is best interpreted using an “effective zero” threshold rather than expecting exact zeros.

---

## 5. Part 2 — Comparing optimizers (GD vs Momentum vs AdaGrad vs Adam)

### 5.1 What was run

Functions:

- Bowl (convex): `f(x,y) = x^2 + 4y^2`
- Camel (non-convex six-hump camel)

Start point:

- `theta0 = (-2.0, -1.5)`

Plots:

- Bowl: function value vs iteration:
  - `plots/part_2_-_bowl_function_value_vs_iteration.png`
- Bowl: trajectories:
  - `plots/part_2_-_bowl_trajectories.png`
- Camel: function value vs iteration:
  - `plots/part_2_-_camel_function_value_vs_iteration.png`
- Camel: trajectories:
  - `plots/part_2_-_camel_trajectories.png`

### 5.2 Findings (2–3 paragraphs)

On the **convex bowl**, all four methods converge quickly to the global minimum at \((0,0)\): the objective value drops to ~0 within the first few dozen iterations. The differences are mainly in the *optimization path*. Plain GD follows a smooth, direct path toward the center. Momentum reaches the optimum quickly but overshoots and oscillates around it (visible as looping trajectories), which is typical when inertia is strong relative to the curvature. AdaGrad and Adam also converge fast; their step sizes effectively shrink near the optimum, producing compact trajectories near the minimum.

On the **six-hump camel** function, optimizer behavior diverges much more. Starting from \((-2,-1.5)\), GD, AdaGrad, and Adam move into a basin in the lower-left region and then plateau at a higher objective value, indicating they got trapped in a local region for the chosen hyperparameters. Momentum is the outlier in this run: it travels further across the landscape and reaches a substantially lower function value, with a trajectory that crosses multiple contour regions before settling. This highlights a key non-convex reality: “good on convex” does not guarantee “good on non-convex,” and whether an optimizer reaches a global minimum can depend on both the starting point and the update dynamics.

Hyperparameters also **do not transfer cleanly** between the two functions. Learning rates that are safe on the bowl can produce oscillations (notably with Momentum) or fail to escape shallow regions on camel. Compared to GD, Momentum can carry enough inertia to traverse flat/shallow regions and sometimes escape local traps, but it can also be oscillatory. AdaGrad adapts learning rates per coordinate, helping in ill-conditioned geometry but often becoming conservative over time. Adam combines momentum with adaptive scaling, which usually stabilizes optimization, but (as seen here) can still converge to a suboptimal basin on a non-convex surface.

---

## 6. Artifacts (where to look)

- All plots: `plots/`
- How to run: `HOW_TO_RUN.md`
- Main script: `llm_architectures_hometask_1.py`

