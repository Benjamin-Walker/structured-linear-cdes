import torch

vocab_size = 12


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Repetition task."""

    length = generator.randint(min_length, max_length)

    if length % 2 == 1:
        length += 1

    sequence = [generator.randint(1, vocab_size - 2) for _ in range(length // 2)]
    target_sequence = sequence + sequence
    sequence += [11]

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'repetition' task."""
    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(input_sequence, dtype=torch.long)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)

    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[input_tensor.shape[0] - 1 :] = True

    return input_tensor, target_tensor, mask
