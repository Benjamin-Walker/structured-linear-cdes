import torch

vocab_size = 11


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Missing Duplicate task."""
    length = generator.randint(min_length, max_length)  # Random length of the sequence
    if length % 2 == 1:
        length += 1
    tokens = [
        generator.randint(1, vocab_size - 2) for _ in range(length // 2)
    ]  # Generate a sequence of tokens

    duplicate = tokens.copy()
    masked_index = generator.randint(0, length // 2 - 1)
    missing_token = duplicate[masked_index]
    duplicate[masked_index] = 10

    return (tokens + duplicate), missing_token


def preprocess_data(sample):
    """Preprocess function for the 'missing_duplicate' task."""
    data, missing_token = sample

    input_tensor = torch.tensor(data, dtype=torch.long)
    index = (input_tensor == 10).nonzero().item()

    target_tensor = torch.zeros(input_tensor.shape, dtype=torch.long)
    target_tensor[index] = missing_token

    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[index] = True

    return input_tensor, target_tensor, mask
