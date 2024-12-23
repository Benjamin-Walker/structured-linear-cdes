import torch

vocab_size = 12


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Odds First task."""

    length = generator.randint(min_length, max_length)

    if length % 2 == 1:
        length += 1

    sequence = [generator.randint(1, vocab_size - 2) for _ in range(length // 2)]

    odd_index_tokens = [sequence[i] for i in range(0, length // 2, 2)]
    even_index_tokens = [sequence[i] for i in range(1, length // 2, 2)]

    sequence = sequence + [11]

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
