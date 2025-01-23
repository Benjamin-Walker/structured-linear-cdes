import torch

vocab_size = 3


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Even Pairs task using PyTorch."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    # Generate sequence length
    length = torch.randint(min_length, max_length + 1, (1,)).item()

    # Generate random binary string of 'a' and 'b'
    binary_string = [
        "a" if torch.randint(0, 2, (1,)).item() == 0 else "b" for _ in range(length)
    ]

    # Determine target based on first and last character equality
    target = "a" if binary_string[0] == binary_string[-1] else "b"

    return "".join(binary_string), target


def preprocess_data(sample):
    """Preprocess function for the 'even_pairs' task."""
    binary_string, target = sample

    char_to_index = {"a": 1, "b": 2}  # Index mapping for input tokens

    input_tensor = torch.tensor(
        [char_to_index[char] for char in binary_string], dtype=torch.long
    )
    target_tensor = torch.zeros_like(input_tensor)
    target_tensor[-1] = 1 if target == "a" else 2

    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
