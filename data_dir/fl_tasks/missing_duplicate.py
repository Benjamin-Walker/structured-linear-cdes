import torch

num_elements = 2  # Number of unique elements
vocab_size = num_elements + 2


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Missing Duplicate task."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = torch.randint(min_length, max_length + 1, (1,)).item()

    if length % 2 == 1:
        length += 1
    tokens = [
        torch.randint(1, num_elements + 1, (1,)).item() for _ in range(length // 2)
    ]

    duplicate = tokens.copy()
    masked_index = torch.randint(0, length // 2, (1,)).item()
    missing_token = duplicate[masked_index]
    duplicate[masked_index] = vocab_size - 1

    return (tokens + duplicate), missing_token


def preprocess_data(sample):
    """Preprocess function for the 'missing_duplicate' task."""
    data, missing_token = sample

    input_tensor = torch.tensor(data, dtype=torch.long)
    index = (input_tensor == vocab_size - 1).nonzero().item()

    target_tensor = torch.zeros(input_tensor.shape, dtype=torch.long)
    target_tensor[index] = missing_token

    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[index] = True

    return input_tensor, target_tensor, mask
