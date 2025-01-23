import torch

vocab_size = 3


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Parity task using PyTorch."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    # Generate sequence length
    length = torch.randint(min_length, max_length + 1, (1,)).item()

    # Generate random sequence of 'a' and 'b'
    sequence = [
        "a" if torch.randint(0, 2, (1,)).item() == 0 else "b" for _ in range(length)
    ]

    # Determine target based on parity of 'b' tokens
    num_b = sum(1 for token in sequence if token == "b")
    target = "a" if num_b % 2 == 0 else "b"

    return sequence, target


def preprocess_data(sample):
    """Preprocess function for the 'parity' task."""
    input_sequence, target = sample

    char_to_index = {"a": 1, "b": 2}  # Index mapping for input tokens

    input_tensor = torch.tensor(
        [char_to_index[char] for char in input_sequence], dtype=torch.long
    )
    target_tensor = torch.zeros(input_tensor.shape[0], dtype=torch.long)
    target_tensor[-1] = 1 if target == "a" else 2

    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
