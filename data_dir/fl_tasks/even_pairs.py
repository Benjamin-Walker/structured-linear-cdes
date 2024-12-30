import torch

vocab_size = 3


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Even Pairs task."""

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = generator.randint(min_length, max_length)
    binary_string = [generator.choice(["a", "b"]) for _ in range(length)]
    target = "a" if binary_string[0] == binary_string[-1] else "b"
    return "".join(binary_string), target


def preprocess_data(sample):
    """Preprocess function for the 'even_pairs' task."""
    binary_string, target = sample
    input_tensor = torch.tensor(
        [1 if char == "a" else 2 for char in binary_string], dtype=torch.long
    )
    target_tensor = torch.zeros_like(input_tensor)
    target_tensor[-1] = 1 if target == "a" else 2
    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True
    return input_tensor, target_tensor, mask
