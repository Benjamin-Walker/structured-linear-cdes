import math

import torch

vocab_size = 4


def binary_sqrt(sequence):
    num = int(
        "".join(map(str, [sequence[i] - 1 for i in range(len(sequence) - 1, -1, -1)])),
        2,
    )
    int_root = math.floor(math.sqrt(num))
    bin_representation = list(bin(int_root)[2:])
    solution = [
        int(bin_representation[i], 2) + 1
        for i in range(len(bin_representation) - 1, -1, -1)
    ]

    return solution


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the binary (little-Endian) Computer Sqrt task."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = torch.randint(min_length, max_length + 1, (1,)).item()

    if length % 6 == 0:
        length -= 1
    elif length % 6 == 3:
        length -= 2

    sequence = [torch.randint(1, 3, (1,)).item() for _ in range(2 * length // 3 - 1)]

    # As representation in little-Endian, make sure final number is 1 (represented as 2)
    sequence.append(2)

    solution = binary_sqrt(sequence)
    target_sequence = (
        [0] * len(sequence) + solution + [0] * (length - len(sequence) - len(solution))
    )

    # Append ACT token
    sequence.append(3)

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'compute_sqrt' task."""

    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(input_sequence, dtype=torch.long)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)
    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[len(input_sequence) - 1 :] = True

    return input_tensor, target_tensor, mask
