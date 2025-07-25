import importlib
import random

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


class A5Dataset(Dataset):
    def __init__(self, length, padding_length=None):
        # Load data from CSV file
        df = pd.read_csv(f"data_dir/illusion/A5={length}.csv")

        # Convert the input and target columns to numpy arrays
        input_array = df["input"].str.split(" ", expand=True).astype(int).to_numpy()
        target_array = df["target"].str.split(" ", expand=True).astype(int).to_numpy()

        # Store data as torch tensors
        self.data = torch.tensor(input_array, dtype=torch.long)
        self.labels = torch.tensor(target_array, dtype=torch.long)
        self.original_length = length
        # If no padding_length is provided, use the original length
        self.padding_length = padding_length if padding_length is not None else length
        self.data_dim = 60
        self.label_dim = 60

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the original sequences
        input_seq = self.data[idx]
        target_seq = self.labels[idx]
        current_length = input_seq.size(0)

        # If the padding length is greater than the original, pad; otherwise, truncate.
        if current_length < self.padding_length:
            pad_size = self.padding_length - current_length
            input_seq = torch.cat(
                [input_seq, torch.zeros(pad_size, dtype=input_seq.dtype)]
            )
            target_seq = torch.cat(
                [target_seq, torch.zeros(pad_size, dtype=target_seq.dtype)]
            )
            # Mask: 1 for data, 0 for padded tokens
            mask = torch.cat(
                [
                    torch.ones(current_length, dtype=torch.bool),
                    torch.zeros(pad_size, dtype=torch.bool),
                ]
            )
        elif current_length > self.padding_length:
            input_seq = input_seq[: self.padding_length]
            target_seq = target_seq[: self.padding_length]
            mask = torch.ones(self.padding_length, dtype=torch.bool)
        else:
            mask = torch.ones(self.padding_length, dtype=torch.bool)

        return input_seq, target_seq, mask


def create_a5_dataloaders(
    length, padding_length=None, train_split=0.8, batch_size=32, seed=1234, shuffle=True
):
    # Initialize the dataset with the new padding_length parameter
    dataset = A5Dataset(length, padding_length)
    data_dim = dataset.data_dim
    label_dim = dataset.label_dim

    # Create DataLoader for training and testing
    if train_split < 1:
        train_size = int(train_split * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = None

    return train_loader, test_loader, data_dim, label_dim


class FormalLanguageDataset(Dataset):
    def __init__(
        self,
        task="even_pairs",
        num_samples=1000,
        min_length=3,
        max_length=10,
        seed=1234,
    ):
        """
        Formal Language Dataset capable of generating samples dynamically.

        Args:
            task (str): The name of the formal language task to generate. Corresponds to a script in the 'tasks' directory.
            num_samples (int): Number of samples in the dataset.
            min_length (int): Minimum length of the sequences.
            max_length (int): Maximum length of the sequences.
            seed (int): Seed for random number generator.
        """
        self.task = task
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        random.seed(seed)
        self.seeds = [random.randint(0, 2**32 - 1) for _ in range(num_samples)]
        self.generate_sample, self.preprocess, self.data_dim, self.label_dim = (
            self._load_task_module()
        )

    def _load_task_module(self):
        """Dynamically loads the task module and retrieves the generate_sample and preprocess_data functions."""
        try:
            # Construct the module name
            module_name = f"data_dir.fl_tasks.{self.task}"
            # Dynamically import the module
            task_module = importlib.import_module(module_name)
            # Check if the module has the required functions
            if hasattr(task_module, "generate_sample") and hasattr(
                task_module, "preprocess_data"
            ):
                # Retrieve the generate_sample function
                generate_sample = task_module.generate_sample
                # Retrieve the preprocess function
                preprocess_data = task_module.preprocess_data
                # Retrieve the data and label dimensions
                data_dim = task_module.vocab_size
                label_dim = task_module.vocab_size
                return generate_sample, preprocess_data, data_dim, label_dim
            else:
                raise ValueError(
                    f"Module '{module_name}' must have 'generate_sample' and 'preprocess_data' functions."
                )
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Task '{self.task}' does not exist. Please provide a valid task name."
            ) from e

    def __len__(self):
        """Returns the total number of samples as defined by num_samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """Generates a sample on the fly based on the task-specific function."""
        # Generate a single data sample
        sample = self.generate_sample(
            self.min_length, self.max_length, seed=self.seeds[idx]
        )
        # Preprocess the sample
        return self.preprocess(sample)


def collate_fn(batch, padding_length=None):
    """
    Collate function to handle padding of variable-length sequences with a non-zero padding value.
    """
    inputs, targets, masks = zip(*batch)
    inputs = list(inputs)
    targets = list(targets)
    masks = list(masks)

    if padding_length is not None:
        inputs.append(torch.zeros(padding_length, dtype=torch.long))
        targets.append(torch.zeros(padding_length, dtype=torch.long))
        masks.append(torch.zeros(padding_length, dtype=torch.bool))

    # Pad sequences to the same length with padding value 0
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    pred_masks = pad_sequence(masks, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    if padding_length is not None:
        padded_inputs = padded_inputs[:-1]
        targets = targets[:-1]
        pred_masks = pred_masks[:-1]

    return padded_inputs, targets, pred_masks


def create_fl_dataloaders(
    task,
    num_samples,
    min_length,
    max_length,
    batch_size,
    padding_length=None,
    train_split=0.8,
    seed=1234,
):
    """
    Creates DataLoader objects for the Formal Language Dataset.

    Args:
        task (str): The name of the formal language task to generate.
        num_samples (int): Number of samples in the dataset.
        min_length (int): Minimum length of the sequences.
        max_length (int): Maximum length of the sequences.
        batch_size (int): Batch size for the DataLoader.
        padding_length (int): Length to pad the sequences to. If None, sequences are padded to the maximum length in
                                the batch.
        train_split (float): Fraction of the dataset to use for training.
        seed (int): Seed for the random number generator.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    # Initialize the dataset
    dataset = FormalLanguageDataset(
        task=task,
        num_samples=num_samples,
        min_length=min_length,
        max_length=max_length,
        seed=seed,
    )

    def col_fn(batch):
        return collate_fn(batch, padding_length=padding_length)

    data_dim = dataset.data_dim
    label_dim = dataset.label_dim

    # Create DataLoader for training and testing
    if train_split < 1:
        # Determine the train/test split
        train_size = int(train_split * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=col_fn,
            num_workers=32,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=col_fn,
            num_workers=32,
        )
    else:
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=col_fn
        )
        test_loader = None

    return train_loader, test_loader, data_dim, label_dim
