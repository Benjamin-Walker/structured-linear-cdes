import torch

vocab_size = 128


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Set task."""

    length = generator.randint(min_length, max_length)

    if length % 2 == 1:
        length += 1

    sequence = [generator.randint(1, vocab_size - 2) for _ in range(length // 2)]

    seen = set()
    target_sequence = []

    for token in sequence:
        if token not in seen:
            target_sequence.append(token)
            seen.add(token)

    target_sequence = [0] * len(sequence) + target_sequence
    sequence += [127]

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'set' task."""
    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(input_sequence, dtype=torch.long)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)

    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[len(input_sequence) - 1 :] = True

    return input_tensor, target_tensor, mask
