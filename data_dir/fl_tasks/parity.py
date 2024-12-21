import torch

vocab_size = 3


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Parity task."""

    length = generator.randint(min_length, max_length)

    sequence = [generator.choice(["a", "b"]) for _ in range(length)]

    num_b = sum(1 for token in sequence if token == "b")

    target = "a" if num_b % 2 == 0 else "b"

    return sequence, target


def preprocess_data(sample):
    """Preprocess function for the 'parity' task."""
    input_sequence, target = sample

    input_tensor = torch.tensor(
        [1 if token == "a" else 2 for token in input_sequence], dtype=torch.long
    )
    target_tensor = torch.zeros(input_tensor.shape[0], dtype=torch.long)
    target_tensor[-1] = 1 if target == "a" else 2

    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
