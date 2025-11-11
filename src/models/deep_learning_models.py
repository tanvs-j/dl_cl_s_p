"""
Deep Learning Models for Seizure Prediction.
Implements CNN, LSTM, CNN-LSTM hybrid, and Transformer-based architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for deep learning models."""
    # Input dimensions
    num_channels: int = 18
    sampling_rate: int = 256
    sequence_length: int = 512  # 2 seconds at 256 Hz
    
    # Model architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-5
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG signals."""
    
    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            eeg_data: EEG signals [num_samples, channels, time_steps]
            labels: Binary labels [num_samples]
        """
        self.data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CNN1D_EEG(nn.Module):
    """1D Convolutional Neural Network for EEG classification."""
    
    def __init__(self, config: ModelConfig):
        """Initialize CNN model."""
        super(CNN1D_EEG, self).__init__()
        
        self.config = config
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(config.num_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate size after convolutions
        conv_output_size = config.sequence_length // 8  # 3 pooling layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * conv_output_size, 256)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(256, 2)  # Binary classification
        
        logger.info(f"Initialized CNN1D model with {self.count_parameters()} parameters")
    
    def forward(self, x):
        """Forward pass."""
        # x shape: [batch, channels, time]
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTM_EEG(nn.Module):
    """LSTM-based model for temporal EEG analysis."""
    
    def __init__(self, config: ModelConfig):
        """Initialize LSTM model."""
        super(LSTM_EEG, self).__init__()
        
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.num_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(config.hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(config.hidden_size * 2, 128)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(128, 2)
        
        logger.info(f"Initialized LSTM model with {self.count_parameters()} parameters")
    
    def forward(self, x):
        """Forward pass with attention."""
        # x shape: [batch, channels, time]
        # Transpose for LSTM: [batch, time, channels]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: [batch, time, hidden_size * 2]
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights shape: [batch, time, 1]
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context shape: [batch, hidden_size * 2]
        
        # Fully connected
        x = self.fc1(context)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN_LSTM_Hybrid(nn.Module):
    """Hybrid CNN-LSTM model combining spatial and temporal features."""
    
    def __init__(self, config: ModelConfig):
        """Initialize hybrid model."""
        super(CNN_LSTM_Hybrid, self).__init__()
        
        self.config = config
        
        # CNN for spatial feature extraction
        self.conv1 = nn.Conv1d(config.num_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        # LSTM for temporal feature extraction
        temporal_length = config.sequence_length // 4  # After 2 pooling layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier
        self.fc1 = nn.Linear(config.hidden_size * 2, 128)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(128, 2)
        
        logger.info(f"Initialized CNN-LSTM Hybrid model with {self.count_parameters()} parameters")
    
    def forward(self, x):
        """Forward pass."""
        # x shape: [batch, channels, time]
        
        # CNN layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Transpose for LSTM: [batch, time, features]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        # Concatenate forward and backward hidden states
        x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerEEG(nn.Module):
    """Transformer-based model for EEG classification."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Transformer model."""
        super(TransformerEEG, self).__init__()
        
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.num_channels, config.hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_size, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Classifier
        self.fc1 = nn.Linear(config.hidden_size, 128)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(128, 2)
        
        logger.info(f"Initialized Transformer model with {self.count_parameters()} parameters")
    
    def forward(self, x):
        """Forward pass."""
        # x shape: [batch, channels, time]
        # Transpose: [batch, time, channels]
        x = x.transpose(1, 2)
        
        # Project to hidden size
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)
        
        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResNet1D_EEG(nn.Module):
    """ResNet-inspired architecture for EEG classification."""
    
    def __init__(self, config: ModelConfig):
        """Initialize ResNet model."""
        super(ResNet1D_EEG, self).__init__()
        
        self.config = config
        
        # Initial convolution
        self.conv1 = nn.Conv1d(config.num_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Linear(256, 2)
        
        logger.info(f"Initialized ResNet1D model with {self.count_parameters()} parameters")
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Create a residual layer."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Residual block for ResNet."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        """Initialize residual block."""
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        """Forward pass."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


def create_model(model_type: str, config: ModelConfig) -> nn.Module:
    """
    Factory function to create deep learning models.
    
    Args:
        model_type: Type of model ('cnn', 'lstm', 'hybrid', 'transformer', 'resnet')
        config: Model configuration
        
    Returns:
        PyTorch model
    """
    models = {
        'cnn': CNN1D_EEG,
        'lstm': LSTM_EEG,
        'hybrid': CNN_LSTM_Hybrid,
        'transformer': TransformerEEG,
        'resnet': ResNet1D_EEG
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model = models[model_type](config)
    model = model.to(config.device)
    
    logger.info(f"Created {model_type} model on {config.device}")
    
    return model


if __name__ == "__main__":
    # Demo: Create and test all models
    config = ModelConfig()
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.num_channels, config.sequence_length)
    dummy_input = dummy_input.to(config.device)
    
    model_types = ['cnn', 'lstm', 'hybrid', 'transformer', 'resnet']
    
    for model_type in model_types:
        logger.info(f"\nTesting {model_type.upper()} model...")
        
        model = create_model(model_type, config)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"  Input shape: {dummy_input.shape}")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Parameters: {model.count_parameters():,}")
