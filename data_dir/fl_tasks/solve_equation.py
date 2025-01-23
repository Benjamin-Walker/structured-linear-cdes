import torch

from data_dir.fl_tasks.mod_arith_w_brack import (
    generate_sample as generate_sample_mod_arith_w_brack,
)

vocab_size = 14  # [PAD], Numbers 0-4, '+', '-', '*', '(', ')', '=', 'x', [ACT]


def generate_sample(min_length, max_length, seed=None):
    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    if seed is not None:
        torch.manual_seed(seed)
    sample = generate_sample_mod_arith_w_brack(min_length, max_length, seed)

    # Replace a number with 'x'

    sequence = sample[0]
    solution = sample[1]

    x_positions = [i for i, token in enumerate(sequence) if token < 6]
    x_position = x_positions[torch.randint(0, len(x_positions), (1,)).item()]
    eqn_solution = sequence[x_position] - 1
    sequence[x_position] = 12  # Token for 'x' (modulus + 7)
    sequence += [solution, 13]

    return sequence, eqn_solution


def preprocess_data(sample):
    """
    Preprocess function for the modular arithmetic task with an unknown variable.

    Args:
        sample: A tuple (input_sequence, solution).

    Returns:
        A tuple (input_tensor, target_tensor, mask) for PyTorch processing.
    """
    input_sequence, solution = sample
    input_tensor = torch.tensor(input_sequence, dtype=torch.long)

    # Create target tensor and mask
    target_tensor = torch.zeros_like(input_tensor, dtype=torch.long)
    target_tensor[-1] = solution
    mask = torch.zeros_like(input_tensor, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
