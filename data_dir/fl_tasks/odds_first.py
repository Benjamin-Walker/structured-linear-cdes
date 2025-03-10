import torch

num_elements = 2  # Number of unique elements
vocab_size = num_elements + 2


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Odds First task."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = torch.randint(min_length, max_length + 1, (1,)).item()

    if length % 2 == 1:
        length += 1

    sequence = [
        torch.randint(1, num_elements + 1, (1,)).item() for _ in range(length // 2)
    ]

    odd_index_tokens = [sequence[i] for i in range(0, length // 2, 2)]
    even_index_tokens = [sequence[i] for i in range(1, length // 2, 2)]

    sequence.append(vocab_size - 1)

    target_sequence = [0] * (length // 2) + odd_index_tokens + even_index_tokens

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'odds_first' task."""
    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(input_sequence, dtype=torch.long)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)

    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[len(input_sequence) - 1 :] = True

    return input_tensor, target_tensor, mask
