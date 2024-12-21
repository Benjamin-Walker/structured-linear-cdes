from collections import Counter

import torch

vocab_size = 64


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Majority task."""
    length = generator.randint(min_length, max_length)

    tokens = [generator.randint(1, vocab_size - 1) for _ in range(length)]

    counts = Counter(tokens)

    max_count = -1
    majority_token = None

    for element in tokens:
        if counts[element] > max_count:
            max_count = counts[element]
            majority_token = element

    return tokens, majority_token


def preprocess_data(sample):
    """Preprocess function for the Majority task."""
    tokens, majority_token = sample
    input_tensor = torch.tensor(tokens, dtype=torch.long)
    target_tensor = torch.zeros_like(input_tensor)
    target_tensor[-1] = majority_token
    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True
    return input_tensor, target_tensor, mask
