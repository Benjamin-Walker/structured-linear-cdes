import warnings

import torch

vocab_size = 5


def binary_multiply(sequence):
    operator_position = next((i for i, a in enumerate(sequence) if a == 3), None)

    if not operator_position:
        raise ValueError("There should be a value of 3 in the sequence")

    num1 = int(
        "".join(
            map(str, [sequence[i] - 1 for i in range(operator_position - 1, -1, -1)])
        ),
        2,
    )
    num2 = int(
        "".join(
            map(
                str,
                [
                    sequence[i] - 1
                    for i in range(len(sequence) - 1, operator_position, -1)
                ],
            )
        ),
        2,
    )

    bin_representation = list(bin(num1 * num2)[2:])
    solution = [
        int(bin_representation[i], 2) + 1
        for i in range(len(bin_representation) - 1, -1, -1)
    ]

    return solution


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Binary (little-Endian) Multiplication task."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    if min_length < 5:
        min_length = max(5, min_length)
        warnings.warn(
            "sequence for binary multiplication must be at least 5, max(5, min_length) applied"
        )

    length = torch.randint(min_length, max_length + 1, (1,)).item()

    if length % 2 == 1:
        length += 1

    sequence = [torch.randint(1, 3, (1,)).item() for _ in range(length // 2 - 1)]

    # As representation in little-Endian, make sure final number is 1 (represented as 2)
    sequence.append(2)

    operator_position = torch.randint(1, len(sequence) - 1, (1,)).item()
    sequence[operator_position] = 3

    # Fix the prior value before operator to be 1 (represented as 2)
    sequence[operator_position - 1] = 2

    solution = binary_multiply(sequence)
    target_sequence = (
        [0] * len(sequence) + solution + [0] * (len(sequence) - len(solution))
    )
    sequence.append(4)

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'binary_multiplication' task."""

    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(input_sequence, dtype=torch.long)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)
    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[len(input_sequence) - 1 :] = True

    return input_tensor, target_tensor, mask
