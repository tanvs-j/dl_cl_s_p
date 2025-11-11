"""
Continual Learning for Seizure Prediction.
Implements online learning, experience replay, and adaptive model updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import copy
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning."""
    # Memory buffer
    memory_size: int = 1000
    replay_batch_size: int = 16
    
    # Learning parameters
    learning_rate: float = 0.0001
    adaptation_rate: float = 0.01  # For fast adaptation
    
    # Update strategy
    update_frequency: int = 10  # Update every N samples
    replay_frequency: int = 5  # Replay every N updates
    
    # Regularization
    ewc_lambda: float = 1000  # Elastic Weight Consolidation
    distillation_temp: float = 2.0  # Knowledge distillation
    
    # Thresholds
    drift_threshold: float = 0.15  # Concept drift detection
    confidence_threshold: float = 0.8


class ExperienceReplayBuffer:
    """Experience replay buffer for continual learning."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of samples to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
        logger.info(f"Initialized replay buffer with size {max_size}")
    
    def add(self, sample: Tuple[np.ndarray, int]):
        """
        Add sample to buffer.
        
        Args:
            sample: Tuple of (data, label)
        """
        self.buffer.append(sample)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            Batch of data and labels
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        
        data = torch.stack([torch.FloatTensor(s[0]) for s in samples])
        labels = torch.LongTensor([s[1] for s in samples])
        
        return data, labels
    
    def __len__(self):
        return len(self.buffer)
    
    def is_full(self):
        return len(self.buffer) >= self.max_size


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention."""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, device: str = 'cpu'):
        """
        Initialize EWC.
        
        Args:
            model: Neural network model
            dataloader: DataLoader with previous task data
            device: Device to run on
        """
        self.model = model
        self.device = device
        
        # Store important parameters
        self.saved_params = {name: param.clone().detach() 
                            for name, param in model.named_parameters() 
                            if param.requires_grad}
        
        # Compute Fisher Information Matrix
        self.fisher = self._compute_fisher(dataloader)
        
        logger.info("Initialized EWC regularization")
    
    def _compute_fisher(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information Matrix."""
        fisher = {name: torch.zeros_like(param) 
                 for name, param in self.model.named_parameters() 
                 if param.requires_grad}
        
        self.model.eval()
        
        for data, labels in dataloader:
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, labels)
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.pow(2)
        
        # Average over samples
        num_samples = len(dataloader.dataset)
        for name in fisher:
            fisher[name] /= num_samples
        
        return fisher
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty."""
        loss = 0
        
        for name, param in model.named_parameters():
            if name in self.fisher and param.requires_grad:
                loss += (self.fisher[name] * 
                        (param - self.saved_params[name]).pow(2)).sum()
        
        return loss


class ConceptDriftDetector:
    """Detect concept drift in EEG patterns."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.15):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of sliding window for statistics
            threshold: Threshold for drift detection
        """
        self.window_size = window_size
        self.threshold = threshold
        
        self.prediction_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        
        self.baseline_acc = None
        self.drift_detected = False
        
        logger.info(f"Initialized drift detector (window={window_size}, threshold={threshold})")
    
    def update(self, prediction: int, true_label: int, confidence: float):
        """
        Update detector with new prediction.
        
        Args:
            prediction: Model prediction
            true_label: True label
            confidence: Prediction confidence
        """
        is_correct = (prediction == true_label)
        
        self.prediction_history.append(is_correct)
        self.confidence_history.append(confidence)
        
        if len(self.prediction_history) >= self.window_size:
            current_acc = np.mean(self.prediction_history)
            
            if self.baseline_acc is None:
                self.baseline_acc = current_acc
            else:
                # Check for drift
                acc_drop = self.baseline_acc - current_acc
                
                if acc_drop > self.threshold:
                    self.drift_detected = True
                    logger.warning(f"⚠️  Concept drift detected! Accuracy drop: {acc_drop:.2%}")
                else:
                    self.drift_detected = False
    
    def get_status(self) -> Dict:
        """Get drift detector status."""
        if len(self.prediction_history) < self.window_size:
            return {
                'drift_detected': False,
                'current_accuracy': None,
                'baseline_accuracy': None,
                'samples_collected': len(self.prediction_history)
            }
        
        return {
            'drift_detected': self.drift_detected,
            'current_accuracy': np.mean(self.prediction_history),
            'baseline_accuracy': self.baseline_acc,
            'avg_confidence': np.mean(self.confidence_history)
        }
    
    def reset(self):
        """Reset detector."""
        self.baseline_acc = np.mean(self.prediction_history) if len(self.prediction_history) > 0 else None
        self.drift_detected = False


class ContinualLearner:
    """Main continual learning system."""
    
    def __init__(self, model: nn.Module, config: ContinualLearningConfig,
                 device: str = 'cpu'):
        """
        Initialize continual learner.
        
        Args:
            model: Base model to adapt
            config: Continual learning configuration
            device: Device to run on
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Components
        self.replay_buffer = ExperienceReplayBuffer(config.memory_size)
        self.drift_detector = ConceptDriftDetector(threshold=config.drift_threshold)
        
        # Optimizers
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.fast_optimizer = optim.SGD(model.parameters(), lr=config.adaptation_rate)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # EWC components
        self.ewc = None
        
        # Statistics
        self.update_count = 0
        self.total_samples = 0
        
        # Teacher model for knowledge distillation
        self.teacher_model = None
        
        logger.info("Initialized Continual Learner")
        logger.info(f"  Memory size: {config.memory_size}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  Device: {device}")
    
    def online_update(self, data: torch.Tensor, label: torch.Tensor,
                     use_replay: bool = True) -> Dict:
        """
        Perform online model update.
        
        Args:
            data: Input data
            label: True label
            use_replay: Whether to use experience replay
            
        Returns:
            Update metrics
        """
        self.model.train()
        
        data = data.to(self.device)
        label = label.to(self.device)
        
        # Forward pass
        output = self.model(data)
        loss = self.criterion(output, label)
        
        # Add EWC penalty if available
        if self.ewc is not None:
            ewc_loss = self.ewc.penalty(self.model)
            loss += self.config.ewc_lambda * ewc_loss
        
        # Add knowledge distillation loss if teacher available
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_output = self.teacher_model(data)
            
            distillation_loss = self._distillation_loss(
                output, teacher_output, self.config.distillation_temp
            )
            loss += distillation_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update statistics
        self.update_count += 1
        self.total_samples += data.size(0)
        
        # Add to replay buffer
        for i in range(data.size(0)):
            sample = (data[i].cpu().numpy(), label[i].item())
            self.replay_buffer.add(sample)
        
        # Experience replay
        replay_loss = 0
        if use_replay and self.update_count % self.config.replay_frequency == 0:
            replay_loss = self._experience_replay()
        
        # Update drift detector
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            confidence = torch.softmax(output, dim=1).max().item()
            
            for i in range(len(pred)):
                self.drift_detector.update(
                    pred[i].item(), 
                    label[i].item(), 
                    confidence
                )
        
        return {
            'loss': loss.item(),
            'replay_loss': replay_loss,
            'update_count': self.update_count,
            'drift_status': self.drift_detector.get_status()
        }
    
    def _experience_replay(self) -> float:
        """Perform experience replay."""
        if len(self.replay_buffer) < self.config.replay_batch_size:
            return 0.0
        
        # Sample from replay buffer
        replay_data, replay_labels = self.replay_buffer.sample(
            self.config.replay_batch_size
        )
        
        replay_data = replay_data.to(self.device)
        replay_labels = replay_labels.to(self.device)
        
        # Forward pass
        output = self.model(replay_data)
        loss = self.criterion(output, replay_labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _distillation_loss(self, student_output: torch.Tensor,
                          teacher_output: torch.Tensor,
                          temperature: float) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        student_soft = F.softmax(student_output / temperature, dim=1)
        teacher_soft = F.softmax(teacher_output / temperature, dim=1)
        
        loss = F.kl_div(
            torch.log(student_soft),
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return loss
    
    def adapt_to_drift(self, adaptation_data: DataLoader):
        """
        Adapt model when concept drift is detected.
        
        Args:
            adaptation_data: DataLoader with recent data
        """
        logger.info("Adapting model to concept drift...")
        
        # Save current model as teacher
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        
        # Compute EWC Fisher matrix on old data
        if len(self.replay_buffer) > 0:
            old_data, old_labels = self.replay_buffer.sample(
                min(len(self.replay_buffer), 100)
            )
            old_dataset = torch.utils.data.TensorDataset(old_data, old_labels)
            old_loader = DataLoader(old_dataset, batch_size=32)
            
            self.ewc = ElasticWeightConsolidation(
                self.model, old_loader, self.device
            )
        
        # Fine-tune on new data
        self.model.train()
        
        for epoch in range(5):  # Quick adaptation
            for data, labels in adaptation_data:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, labels)
                
                # Add EWC penalty
                if self.ewc is not None:
                    loss += self.config.ewc_lambda * self.ewc.penalty(self.model)
                
                # Add distillation loss
                with torch.no_grad():
                    teacher_output = self.teacher_model(data)
                loss += self._distillation_loss(
                    output, teacher_output, self.config.distillation_temp
                )
                
                # Backward pass
                self.fast_optimizer.zero_grad()
                loss.backward()
                self.fast_optimizer.step()
        
        # Reset drift detector
        self.drift_detector.reset()
        
        logger.success("✓ Model adaptation complete")
    
    def save_checkpoint(self, path: str):
        """Save continual learning checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'replay_buffer': list(self.replay_buffer.buffer),
            'update_count': self.update_count,
            'total_samples': self.total_samples,
            'drift_detector_state': {
                'baseline_acc': self.drift_detector.baseline_acc,
                'prediction_history': list(self.drift_detector.prediction_history),
                'confidence_history': list(self.drift_detector.confidence_history)
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load continual learning checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore replay buffer
        self.replay_buffer.buffer = deque(
            checkpoint['replay_buffer'],
            maxlen=self.config.memory_size
        )
        
        self.update_count = checkpoint['update_count']
        self.total_samples = checkpoint['total_samples']
        
        # Restore drift detector
        drift_state = checkpoint['drift_detector_state']
        self.drift_detector.baseline_acc = drift_state['baseline_acc']
        self.drift_detector.prediction_history = deque(
            drift_state['prediction_history'],
            maxlen=self.drift_detector.window_size
        )
        self.drift_detector.confidence_history = deque(
            drift_state['confidence_history'],
            maxlen=self.drift_detector.window_size
        )
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def get_stats(self) -> Dict:
        """Get learning statistics."""
        return {
            'total_updates': self.update_count,
            'total_samples': self.total_samples,
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_full': self.replay_buffer.is_full(),
            'drift_status': self.drift_detector.get_status()
        }


if __name__ == "__main__":
    # Demo continual learning
    from src.models.deep_learning_models import create_model, ModelConfig
    
    logger.info("Continual Learning Demo")
    logger.info("=" * 60)
    
    # Create model
    model_config = ModelConfig()
    model = create_model('cnn', model_config)
    
    # Create continual learner
    cl_config = ContinualLearningConfig()
    learner = ContinualLearner(model, cl_config, device=model_config.device)
    
    # Simulate online learning
    for i in range(50):
        # Generate dummy data
        data = torch.randn(2, 18, 512)
        labels = torch.randint(0, 2, (2,))
        
        # Online update
        metrics = learner.online_update(data, labels)
        
        if i % 10 == 0:
            logger.info(f"Update {i}: Loss={metrics['loss']:.4f}")
    
    # Print statistics
    stats = learner.get_stats()
    logger.info("\n" + "=" * 60)
    logger.info("Learning Statistics:")
    logger.info(f"  Total updates: {stats['total_updates']}")
    logger.info(f"  Replay buffer: {stats['replay_buffer_size']}/{cl_config.memory_size}")
    logger.info(f"  Drift detected: {stats['drift_status']['drift_detected']}")
