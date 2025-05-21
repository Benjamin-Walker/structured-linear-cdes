import random
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader

from data_dir.dataloaders import (
    FormalLanguageDataset,
    create_fl_dataloaders,
)

mock_samples = [
    ("abc", 0),
    ("defg", 1),
    ("hi", 0),
    ("jklmn", 1),
    ("opqrs", 0),
    ("tuv", 1),
    ("wx", 0),
    ("yz", 1),
]

# Mock data to simulate task-specific data generation
mock_tensors = [
    (torch.tensor([1, 2, 3]), torch.tensor([0]), torch.tensor([0, 0, 1])),
    (torch.tensor([4, 5, 6, 7]), torch.tensor([1]), torch.tensor([0, 1, 0, 0])),
    (torch.tensor([8, 9]), torch.tensor([0]), torch.tensor([0, 1])),
    (
        torch.tensor([10, 11, 12, 13, 14]),
        torch.tensor([1]),
        torch.tensor([0, 0, 1, 0, 0]),
    ),
    (
        torch.tensor([15, 16, 17, 18, 19]),
        torch.tensor([0]),
        torch.tensor([0, 0, 1, 0, 0]),
    ),
    (torch.tensor([20, 21, 22]), torch.tensor([1]), torch.tensor([0, 1, 0])),
    (torch.tensor([23, 24]), torch.tensor([0]), torch.tensor([0, 1])),
    (torch.tensor([25, 26]), torch.tensor([1]), torch.tensor([0, 1])),
]

# Mock parameters
mock_task = "mock_task"
mock_num_samples = 8
mock_min_length = 2
mock_max_length = 5
mock_vocab_size = 26
mock_batch_size = 2


# Define the side effect functions outside the tests to reuse
def generate_sample_side_effect(min_length, max_length, seed=None):
    """Mocked generate_sample function to produce consistent outputs."""
    # Use the generator's state to ensure consistent outputs
    if seed is not None:
        torch.manual_seed(seed)
    index = torch.randint(0, mock_num_samples, (1,)).item()
    return mock_samples[index]


def preprocess_data_side_effect(sample):
    """Mocked preprocess_data function to map samples to their corresponding tensor data."""
    # Find the index in mock_samples to fetch the corresponding tensor data
    index = mock_samples.index(sample)
    return mock_tensors[index]


mock_generate_sample = MagicMock(side_effect=generate_sample_side_effect)
mock_preprocess_data = MagicMock(side_effect=preprocess_data_side_effect)


# Patching the task module's generate_sample and preprocess_data functions
@patch("data_dir.dataloaders.FormalLanguageDataset._load_task_module")
def test_formal_language_dataset_init(mock_load_task_module):
    # Set the return values of the mocked _load_task_module method
    mock_load_task_module.return_value = (
        mock_generate_sample,
        mock_preprocess_data,
        mock_vocab_size,
        mock_vocab_size,
    )

    # Create the dataset
    dataset = FormalLanguageDataset(
        task=mock_task,
        num_samples=mock_num_samples,
        min_length=mock_min_length,
        max_length=mock_max_length,
        seed=0,
    )

    random.seed(0)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(mock_num_samples)]

    # Assertions to check if the dataset is loaded correctly
    assert len(dataset) == mock_num_samples
    for i in range(mock_num_samples):
        torch.manual_seed(seeds[i])
        index = torch.randint(0, mock_num_samples, (1,)).item()
        assert dataset[i] == mock_tensors[index]


@patch("data_dir.dataloaders.FormalLanguageDataset._load_task_module")
def test_create_fl_dataloaders(mock_load_task_module):
    # Set the return values of the mocked _load_task_module method
    mock_load_task_module.return_value = (
        mock_generate_sample,
        mock_preprocess_data,
        mock_vocab_size,
        mock_vocab_size,
    )

    # Create dataloaders
    train_loader, test_loader, data_dim, label_dim = create_fl_dataloaders(
        task=mock_task,
        num_samples=mock_num_samples,
        min_length=mock_min_length,
        max_length=mock_max_length,
        batch_size=mock_batch_size,
        train_split=0.75,
        seed=0,
    )

    # Assertions to check if the dataloaders are created correctly
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert len(train_loader.dataset) == int(0.75 * mock_num_samples)
    assert len(test_loader.dataset) == int(0.25 * mock_num_samples)
    assert data_dim == mock_vocab_size
    assert label_dim == mock_vocab_size

    # Check batch content for train_loader
    for batch in train_loader:
        inputs, targets, pred_masks = batch
        assert inputs.size(0) == mock_batch_size
        assert targets.size(0) == mock_batch_size
        assert pred_masks.size(0) == mock_batch_size

    # Check batch content for test_loader
    for batch in test_loader:
        inputs, targets, pred_masks = batch
        assert inputs.size(0) == mock_batch_size
        assert targets.size(0) == mock_batch_size
        assert pred_masks.size(0) == mock_batch_size


@patch("data_dir.dataloaders.FormalLanguageDataset._load_task_module")
def test_create_fl_dataloaders_no_test_split(mock_load_task_module):
    # Set the return values of the mocked _load_task_module method
    mock_load_task_module.return_value = (
        mock_generate_sample,
        mock_preprocess_data,
        mock_vocab_size,
        mock_vocab_size,
    )

    # Create dataloaders with no test split
    train_loader, test_loader, data_dim, label_dim = create_fl_dataloaders(
        task=mock_task,
        num_samples=mock_num_samples,
        min_length=mock_min_length,
        max_length=mock_max_length,
        batch_size=mock_batch_size,
        train_split=1.0,
        seed=0,
    )

    # Assertions to check if the dataloader is created correctly with no test split
    assert isinstance(train_loader, DataLoader)
    assert test_loader is None
    assert len(train_loader.dataset) == mock_num_samples  # All data in train_loader
    assert data_dim == mock_vocab_size
    assert label_dim == mock_vocab_size

    # Check batch content
    for batch in train_loader:
        inputs, targets, pred_masks = batch
        assert inputs.size(0) == mock_batch_size
        assert targets.size(0) == mock_batch_size
        assert pred_masks.size(0) == mock_batch_size
