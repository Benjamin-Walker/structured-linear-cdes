import torch

vocab_size = 4


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Reverse String task."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = torch.randint(min_length, max_length + 1, (1,)).item()

    if length % 2 == 1:
        length += 1

    sequence = [
        torch.randint(1, vocab_size - 2, (1,)).item() for _ in range(length // 2)
    ]
    target_sequence = sequence + sequence[::-1]
    sequence += [3]

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'reverse_string' task."""
    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(input_sequence, dtype=torch.long)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)

    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[input_tensor.shape[0] - 1 :] = True

    return input_tensor, target_tensor, mask
