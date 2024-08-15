import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class A5Dataset(Dataset):
    def __init__(self, length):
        # Load data from CSV file
        df = pd.read_csv(f"data_dir/illusion/A5={length}.csv")

        # Convert the input and target columns to numpy arrays
        input_array = df["input"].str.split(" ", expand=True).astype(int).to_numpy()
        target_array = df["target"].str.split(" ", expand=True).astype(int).to_numpy()

        # Store data as torch tensors
        self.data = torch.tensor(input_array, dtype=torch.float32)
        self.labels = torch.tensor(target_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_a5_dataloaders(length, train_split=0.8, batch_size=32, shuffle=True):
    # Initialize the dataset
    dataset = A5Dataset(length)

    # Create DataLoader for training and testing
    if train_split < 1:
        # Determine the train/test split
        train_size = int(train_split * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = None

    return train_loader, test_loader
