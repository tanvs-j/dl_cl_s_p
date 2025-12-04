# Networks, Models, and Algorithms Used

This project implements several deep learning architectures for EEG-based seizure prediction and related tasks. Models are implemented mainly in `models/network.py` and `src/models/deep_learning_models.py`, and are trained using PyTorch in `training/train.py`, with optional continual-learning support in `models/ewc.py`.

---

## 1. Core EEG Network Architectures (`models/network.py`)

### 1.1 `EEGNet1D`
- **Type**: 1D Convolutional Neural Network (CNN).
- **Input shape**: `(batch_size, channels, time)`.
- **Architecture**:
  - Stack of 1D convolutional layers with kernel sizes 7, 7, 5, and 3.
  - Each conv layer is followed by Batch Normalization and ReLU.
  - Max pooling layers progressively downsample the temporal dimension.
  - A final **Adaptive Average Pooling** layer reduces the temporal dimension to 1.
  - A fully connected (linear) layer maps features to `num_classes`.
- **Purpose**: Baseline CNN for EEG classification that captures local temporal patterns per channel.

---

### 1.2 `EEGGoogLeNet`
- **Type**: GoogLeNet‑inspired 1D architecture.
- **Key component**: `InceptionModule1D`.
  - Multiple parallel branches with different kernel sizes (1, 3, 5) and a pooling branch.
  - Outputs of branches are concatenated along the channel dimension.
- **Architecture**:
  - Initial 1D convolution + max pooling to reduce temporal resolution.
  - Three stacked `InceptionModule1D` blocks.
  - Adaptive average pooling, flattening, dropout, and a final linear classifier.
- **Purpose**: Capture multi‑scale temporal features (short- and long-range dependencies) in EEG signals.

---

### 1.3 `EEGDenseNet`
- **Type**: DenseNet‑inspired 1D CNN.
- **Related block**: `DenseBlock1D` (conceptual dense block with feature concatenation).
- **Architecture (simplified)**:
  - Initial 1D convolution + batch norm + ReLU + max pooling.
  - Two deeper convolutional blocks (`block1`, `block2`) that progressively increase channels (64 → 128 → 256), each with multiple conv + BN + ReLU layers.
  - Adaptive average pooling + dropout + linear classifier.
- **Purpose**: Encourage feature reuse and deeper representations while keeping the model manageable for 1D EEG.

---

### 1.4 `EEGVGG`
- **Type**: VGG‑style 1D CNN.
- **Architecture**:
  - Sequential convolutional blocks with small kernels (3) and increasing channels:
    - Block 1: 64 channels, two conv layers + BN + ReLU, then max pool.
    - Block 2: 128 channels, two conv layers + BN + ReLU, then max pool.
    - Block 3: 256 channels, three conv layers + BN + ReLU, then max pool.
  - Classifier: Adaptive avg pool → flatten → dropout → linear → ReLU → dropout → final linear.
- **Purpose**: A deeper CNN with many stacked small‑kernel convolutions, good at capturing hierarchical temporal features.

---

### 1.5 `EEGResNet` and `ResidualBlock1D`
- **Type**: ResNet‑inspired 1D CNN.
- **ResidualBlock1D**:
  - Two 1D conv layers with BN and ReLU.
  - Optional skip (shortcut) path with 1x1 conv to match dimensions when stride or channels change.
  - Output: `F.relu(conv2(x) + skip(x))`.
- **EEGResNet architecture**:
  - Initial 1D conv + BN + ReLU + max pooling.
  - Three residual layers combining multiple `ResidualBlock1D` blocks, increasing channels (64 → 128 → 256) and downsampling via stride.
  - Adaptive avg pooling + dropout + linear classifier.
- **Purpose**: Train deeper networks using residual connections, helping gradients flow and avoiding vanishing‑gradient issues.

---

### 1.6 `EEGRNN`
- **Type**: CNN + LSTM (RNN‑enhanced) architecture for EEG.
- **Architecture**:
  - 1D conv feature extractor (two conv layers with BN + ReLU, then max pooling) to learn local temporal/channel patterns.
  - Transpose to `(batch_size, time, features)` and pass through a 2‑layer LSTM.
  - Use the **last LSTM output** as a summary of the sequence.
  - Fully connected classifier with dropout and ReLU.
- **Purpose**: Combine convolutional feature extraction with recurrent modeling of longer temporal dependencies.

---

### 1.7 `DeepCNN1D`
- **Type**: Deeper 1D CNN with modern regularization.
- **Architecture**:
  - Multiple convolutional blocks with increasing channels (32 → 64 → 128 → 256).
  - Each block uses conv + BN + ReLU + max pooling, with increasing dropout rates.
  - Final adaptive avg pooling, flatten, linear → ReLU → dropout → final linear.
- **Purpose**: High‑capacity CNN for complex EEG patterns with stronger regularization (dropout) to reduce overfitting.

---

### 1.8 `create_model` factory (in `models/network.py`)
- **Function**: `create_model(model_name, in_channels, num_classes, **kwargs)`.
- **Supported names**:
  - `'eegnet'`, `'googlenet'`, `'densenet'`, `'vgg'`, `'resnet'`, `'rnn'`, `'deepcnn'`.
- **Usage**:
  - Central factory that instantiates the chosen architecture with the correct number of input channels and output classes.
  - Used in `training/train.py` to construct the model based on CLI arguments (`args.model`).

---

## 2. Deep Learning Models in `src/models/deep_learning_models.py`

This module provides additional architectures and utilities for seizure prediction.

### 2.1 `ModelConfig`
- **Dataclass** holding configuration:
  - Input: `num_channels`, `sampling_rate`, `sequence_length`.
  - Architecture: `hidden_size`, `num_layers`, `dropout`.
  - Training: `batch_size`, `learning_rate`, `num_epochs`, `weight_decay`.
  - Device: `device` (CPU or GPU).
- **Purpose**: Centralizes model and training hyperparameters.

### 2.2 `EEGDataset` (deep_learning_models)
- Wraps EEG data and labels for PyTorch training.
- Expects `eeg_data` as `[num_samples, channels, time_steps]` and `labels` as binary targets.
- Returns `(data, label)` pairs from `__getitem__`.

---

### 2.3 `CNN1D_EEG`
- **Type**: 1D CNN for EEG classification.
- **Architecture**:
  - Three conv blocks (64, 128, 256 channels) with BN + ReLU + max pooling.
  - Feature size scaled by `sequence_length // 8` due to three pooling layers.
  - Flatten → fully connected layer (256 units) → dropout → final classifier (2 classes).
- **Purpose**: Strong baseline CNN model for binary seizure vs. non‑seizure classification.

---

### 2.4 `LSTM_EEG` with Attention
- **Type**: Bidirectional LSTM with attention.
- **Architecture**:
  - Input transposed to `[batch, time, channels]`.
  - Bi‑LSTM with `hidden_size`, `num_layers`, dropout (if `num_layers > 1`).
  - Attention layer computes a scalar weight for each time step and normalizes via softmax.
  - Weighted sum of LSTM outputs forms a context vector.
  - Fully connected layers + dropout → final 2‑class output.
- **Purpose**: Capture long‑range temporal dependencies and highlight the most informative time steps via attention.

---

### 2.5 `CNN_LSTM_Hybrid`
- **Type**: Hybrid CNN + LSTM model.
- **Architecture**:
  - CNN part: two conv blocks (64, 128 channels) with BN + ReLU + max pooling.
  - Output is transposed to `[batch, time, features]` and fed to a Bi‑LSTM.
  - Final representation from the concatenated forward/backward hidden states.
  - Fully connected + dropout → 2‑class classifier.
- **Purpose**: Combine spatial (channel‑wise/temporal locality) and temporal modeling in a single architecture.

---

### 2.6 `TransformerEEG` and `PositionalEncoding`
- **Type**: Transformer‑based model for EEG.
- **Architecture**:
  - Input transpose to `[batch, time, channels]`.
  - Linear `input_projection` maps channels → `hidden_size`.
  - `PositionalEncoding` adds sinusoidal positional embeddings.
  - Multi‑layer Transformer encoder with multi‑head self‑attention.
  - Global average pooling over time.
  - Fully connected + dropout → 2‑class output.
- **Purpose**: Use self‑attention to model long‑range temporal relationships and interactions across time steps.

---

### 2.7 `ResNet1D_EEG` and `ResidualBlock`
- **Type**: ResNet‑like 1D CNN for EEG.
- **Architecture**:
  - Initial conv + BN + ReLU + max pool.
  - Three residual layers built from `ResidualBlock` modules:
    - Each `ResidualBlock` has two conv + BN layers and an identity/shortcut connection.
    - When spatial size or channels change, the shortcut uses a 1x1 conv + BN.
  - Adaptive avg pooling + linear classifier.
- **Purpose**: Deeper CNN with residual connections to maintain gradient flow and capture complex temporal patterns.

---

### 2.8 `create_model` factory (deep_learning_models)
- **Function**: `create_model(model_type, config)`.
- **Supported model types**:
  - `'cnn'`, `'lstm'`, `'hybrid'`, `'transformer'`, `'resnet'`.
- **Behavior**:
  - Instantiates the corresponding model class with `config`.
  - Moves the model to `config.device`.
- **Purpose**: Single entry point to build different model families using a shared configuration.

---

## 3. Training Algorithms and Utilities

### 3.1 Training Loop (`training/train.py`)
- **Dataset**: `EEGDataset` reads EEG recordings from disk, applies preprocessing (`preprocess_for_model`), and returns windows and labels.
- **Collation**: `collate_batch` stacks variable‑length window sets into a single batch of windows, repeating labels per window.
- **Model creation**: Uses `create_model(args.model, in_channels, num_classes=2)`.
- **Loss function**: `nn.CrossEntropyLoss` for binary classification in logits space.
- **Optimizers** (selected via CLI `args.optimizer`):
  - Adam: `optim.Adam`, with configurable learning rate and weight decay.
  - SGD: `optim.SGD`, with momentum and weight decay.
  - AdamW: `optim.AdamW`, decoupled weight decay.
- **Learning‑rate schedulers** (via `args.scheduler`):
  - `ReduceLROnPlateau`: reduces LR when validation accuracy plateaus.
  - `CosineAnnealingLR`: cosine schedule over epochs.
- **Regularization and stabilization**:
  - **Gradient clipping** using `torch.nn.utils.clip_grad_norm_` if `args.grad_clip > 0`.
  - **Weight decay** via optimizer.
  - **Dropout** layers inside networks.
- **Early stopping**:
  - Tracks best accuracy; if it does not improve for `args.patience` epochs, training stops early.
  - Saves the best model checkpoint with state dict, model hyperparameters, optimizer state, epoch, accuracy, and loss.

---

## 4. Continual Learning Algorithm: Elastic Weight Consolidation (`models/ewc.py`)

### 4.1 `EWC` (Elastic Weight Consolidation)
- **Goal**: Mitigate catastrophic forgetting when training a model on **multiple sequential tasks**.
- **Key ideas**:
  - Estimate the **Fisher Information Matrix** for each parameter to measure its importance for previous tasks.
  - After learning a task, store:
    - `fisher_information` per parameter.
    - `optimal_params` (parameter values after training that task).
  - During new task training, add a quadratic penalty if parameters move away from their previous optimal values, scaled by importance.
- **Important methods**:
  - `compute_fisher_information(dataloader, criterion, device, num_samples)`:
    - Runs the model on samples, computes gradients of the loss, and accumulates squared gradients as an approximation of Fisher information.
  - `ewc_loss()` / `get_regularization_loss()`:
    - Returns the EWC regularization term: \(\lambda/2 * \sum F_i (\theta_i - \theta_i^*)^2\).
  - `consolidate_task(...)`:
    - Finalizes a task: computes Fisher and stores it with the corresponding optimal parameters.

### 4.2 `EWCTrainer`
- **Wraps** a model and an `EWC` instance to train across multiple tasks.
- **`train_task(...)`**:
  - Trains for several epochs on a single task.
  - Total loss = task loss (e.g., cross‑entropy) + EWC regularization loss from all previous tasks.
  - Optionally calls `consolidate_task` at the end to record Fisher and optimal parameters.
- **`evaluate_all_tasks(...)`**:
  - Evaluates performance on all tasks’ dataloaders.
- **`plot_learning_progress(...)` and `get_summary_stats()`**:
  - Provide visualization and summary statistics of continual learning performance.

---

## 5. Summary

- **Networks / Models**:
  - Multiple 1D CNN, VGG‑like, ResNet‑like, DenseNet‑like, GoogLeNet‑like, RNN‑based, CNN‑LSTM, and Transformer architectures tailored for EEG signals.
- **Algorithms**:
  - Supervised classification with cross‑entropy.
  - Optimization with Adam, SGD, or AdamW; LR scheduling via ReduceLROnPlateau or CosineAnnealingLR.
  - Regularization techniques: dropout, batch normalization, weight decay, gradient clipping, and **Elastic Weight Consolidation (EWC)** for continual learning.
- **Usage**:
  - Models are created via `create_model` factory functions and trained in `training/train.py` or via the EWC framework in `models/ewc.py`.
