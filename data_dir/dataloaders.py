import importlib
import random

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


class A5Dataset(Dataset):
    def __init__(self, length):
        # Load data from CSV file
        df = pd.read_csv(f"data_dir/illusion/A5={length}.csv")

        # Convert the input and target columns to numpy arrays
        input_array = df["input"].str.split(" ", expand=True).astype(int).to_numpy()
        target_array = df["target"].str.split(" ", expand=True).astype(int).to_numpy()

        # Store data as torch tensors
        self.data = torch.tensor(input_array, dtype=torch.long)
        self.labels = torch.tensor(target_array, dtype=torch.long)
        self.length = length
        self.data_dim = 60
        self.label_dim = 60

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.labels[idx],
            torch.ones((self.length,), dtype=torch.bool),
        )


def create_a5_dataloaders(
    length, train_split=0.8, batch_size=32, seed=1234, shuffle=True
):
    # Initialize the dataset
    dataset = A5Dataset(length)

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
        generator_seed=1234,
    ):
        """
        Formal Language Dataset capable of generating samples dynamically.

        Args:
            task (str): The name of the formal language task to generate. Corresponds to a script in the 'tasks' directory.
            num_samples (int): Number of samples in the dataset.
            min_length (int): Minimum length of the sequences.
            max_length (int): Maximum length of the sequences.
            generator_seed (int): Seed for random number generator.
        """
        self.task = task
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.generator_seed = generator_seed
        self.generator = random.Random(self.generator_seed)
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
        # Ensure reproducibility by resetting the seed for each sample generation
        self.generator.seed(self.generator_seed + idx)
        # Generate a single data sample
        sample = self.generate_sample(self.min_length, self.max_length, self.generator)
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
        task=task, num_samples=num_samples, min_length=min_length, max_length=max_length
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
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=col_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=col_fn
        )
    else:
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=col_fn
        )
        test_loader = None

    return train_loader, test_loader, data_dim, label_dim
