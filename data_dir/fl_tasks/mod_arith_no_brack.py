import torch

modulus = 5
vocab_size = modulus + 5


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Modular Arithmetic task with BIDMAS order of operations."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    original_length = torch.randint(min_length, max_length + 1, (1,)).item()

    if original_length % 2 == 1:
        length = original_length + 1
    else:
        length = original_length

    res = [None] * length

    # Fill in numbers
    for i in range(0, length, 2):
        res[i] = torch.randint(5, 10, (1,)).item()

    # Fill in operators
    for i in range(1, length - 1, 2):
        res[i] = torch.randint(1, 4, (1,)).item()

    # Set the '=' operator at the second last position
    res[-1] = 4

    # First pass: resolve multiplication
    values = [res[0] - 5]  # Initialize with the first number
    operators = []  # To hold the '+' and '-' operators

    for i in range(1, length - 2, 2):
        op = res[i]
        num = res[i + 1] - 5

        if op == 3:  # Multiplication ('*')
            # Apply multiplication directly to the last number in values
            values[-1] *= num
        else:
            # Store addition or subtraction operation for later
            values.append(num)
            operators.append(op)

    # Second pass: process addition and subtraction
    total_val = values[0]

    for j in range(len(operators)):
        if operators[j] == 1:  # Addition ('+')
            total_val += values[j + 1]
        elif operators[j] == 2:  # Subtraction ('-')
            total_val -= values[j + 1]

    # Calculate the final result modulo max_num (5 in this context)
    target = total_val % 5 + 5

    return res, target


def preprocess_data(sample):
    """Preprocess function for the 'modular_arithmetic' task."""
    input, target = sample
    input_tensor = torch.tensor(input, dtype=torch.long)
    target_tensor = torch.zeros_like(input_tensor)
    target_tensor[-1] = target
    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True
    return input_tensor, target_tensor, mask
