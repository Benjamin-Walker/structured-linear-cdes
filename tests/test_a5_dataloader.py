from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_dir.dataloaders import A5Dataset, create_a5_dataloaders

# Mock data for testing
mock_data = {
    "input": [
        "1 2 3",
        "4 5 6",
        "7 8 9",
        "10 11 12",
        "13 14 15",
        "16 17 18",
        "19 20 21",
        "22 23 24",
    ],
    "target": ["0 1", "1 0", "0 1", "1 0", "0 1", "1 0", "0 1", "1 0"],
}

# Convert mock data to DataFrame
mock_df = pd.DataFrame(mock_data)

# Mock dataset parameters
mock_length = 3
mock_target_size = 2
mock_number_of_samples = 8
mock_batch_size = 2


@patch("pandas.read_csv")
def test_a5dataset_init(mock_read_csv):
    # Mock pandas read_csv
    mock_read_csv.return_value = mock_df

    # Create dataset
    dataset = A5Dataset(length=mock_length)

    # Assertions to check if the dataset is loaded correctly
    assert len(dataset) == mock_number_of_samples
    assert torch.equal(dataset[0][0], torch.tensor([1, 2, 3], dtype=torch.float32))
    assert torch.equal(dataset[0][1], torch.tensor([0, 1], dtype=torch.float32))


@patch("pandas.read_csv")
def test_create_a5_dataloaders(mock_read_csv):
    # Mock pandas read_csv
    mock_read_csv.return_value = mock_df

    # Create dataloaders
    train_loader, test_loader, data_dim, label_dim = create_a5_dataloaders(
        length=mock_length, train_split=0.75, batch_size=mock_batch_size
    )

    # Assertions to check if the dataloaders are created correctly
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert len(train_loader.dataset) == int(0.75 * mock_number_of_samples)
    assert len(test_loader.dataset) == int(0.25 * mock_number_of_samples)
    assert data_dim == 60
    assert label_dim == 60

    # Check batch content
    train_indices = set()
    test_indices = set()

    for batch in train_loader:
        inputs, targets, masks = batch
        assert inputs.shape == (mock_batch_size, mock_length)
        assert targets.shape == (mock_batch_size, mock_target_size)
        assert masks.shape == (mock_batch_size, mock_length)
        for input_tensor in inputs:
            train_indices.add(tuple(input_tensor.tolist()))

    for batch in test_loader:
        inputs, targets, masks = batch
        assert inputs.shape == (mock_batch_size, mock_length)
        assert targets.shape == (mock_batch_size, mock_target_size)
        assert masks.shape == (mock_batch_size, mock_length)
        for input_tensor in inputs:
            test_indices.add(tuple(input_tensor.tolist()))

    # Check that there are no duplicates between train and test sets
    assert train_indices.isdisjoint(test_indices), "Train and test sets overlap!"


@patch("pandas.read_csv")
def test_create_a5_dataloaders_no_test_split(mock_read_csv):
    # Mock pandas read_csv
    mock_read_csv.return_value = mock_df

    # Create dataloaders with no test split
    train_loader, test_loader, data_dim, label_dim = create_a5_dataloaders(
        length=mock_length, train_split=1.0, batch_size=mock_batch_size
    )

    # Assertions to check if the dataloader is created correctly with no test split
    assert isinstance(train_loader, DataLoader)
    assert test_loader is None
    assert (
        len(train_loader.dataset) == mock_number_of_samples
    )  # All data in train_loader
    assert data_dim == 60
    assert label_dim == 60

    # Check batch content
    for batch in train_loader:
        inputs, targets, masks = batch
        assert inputs.shape == (mock_batch_size, mock_length)
        assert targets.shape == (mock_batch_size, mock_target_size)
        assert masks.shape == (mock_batch_size, mock_length)
