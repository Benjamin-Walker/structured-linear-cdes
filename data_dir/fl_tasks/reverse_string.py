import torch

vocab_size = 4


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Reverse String task."""

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = generator.randint(min_length, max_length)

    if length % 2 == 1:
        length += 1

    sequence = [generator.randint(1, vocab_size - 2) for _ in range(length // 2)]
    target_sequence = sequence + sequence[::-1]
    sequence += [11]

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'reverse_string' task."""
    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(input_sequence, dtype=torch.long)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)

    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[input_tensor.shape[0] - 1 :] = True

    return input_tensor, target_tensor, mask
