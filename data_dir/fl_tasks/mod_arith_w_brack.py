import torch

modulus = 5
vocab_size = modulus + 7


def generate_sample(min_length, max_length, seed=None):
    """
    Generates a single sample for the Modular Arithmetic task with brackets using PyTorch, with an optional fixed seed.
    """
    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    if min_length < 2:
        raise ValueError(
            "min_length must be at least 2, else only contain terminal token"
        )

    def gen_terminal():
        """Generates a random terminal value."""
        value = torch.randint(0, modulus, (1,)).item()
        return str(value), value

    def generate_expression(length):
        """Recursively generates an expression of the given length."""
        if length < 1:
            raise ValueError(f"Length must be at least 1, got {length}.")
        if length == 1:
            term_str, term_val = gen_terminal()
            return term_str, term_val
        elif length == 2:
            term_str, term_val = gen_terminal()
            return f"-{term_str}", (-term_val) % modulus
        elif length == 3:
            term_str, term_val = gen_terminal()
            return f"({term_str})", term_val % modulus
        elif length == 4:
            term_str, term_val = gen_terminal()
            return f"(-{term_str})", (-term_val) % modulus

        # Split length into left and right parts
        left_length = torch.randint(1, length - 3, (1,)).item()
        right_length = length - (left_length + 3)
        assert left_length >= 1 and right_length >= 1

        left_str, left_val = generate_expression(left_length)
        right_str, right_val = generate_expression(right_length)

        # Sample an operator
        op = torch.randint(0, 3, (1,)).item()
        if op == 0:  # Addition
            return f"({left_str}+{right_str})", (left_val + right_val) % modulus
        elif op == 1:  # Subtraction
            return f"({left_str}-{right_str})", (left_val - right_val) % modulus
        else:  # Multiplication
            return f"({left_str}*{right_str})", (left_val * right_val) % modulus

    # Generate an expression of random length between min_length and max_length
    length = torch.randint(min_length, max_length + 1, (1,)).item()
    expression_str, result = generate_expression(length)

    # Convert expression to numerical representation
    vocab = {
        "+": modulus + 1,
        "-": modulus + 2,
        "*": modulus + 3,
        "(": modulus + 4,
        ")": modulus + 5,
    }
    for i in range(modulus):
        vocab[str(i)] = i + 1

    input_sequence = [vocab[char] for char in expression_str]

    # Add the '=' operator at the end
    input_sequence.append(modulus + 6)

    target_value = vocab[str(result)]
    return input_sequence, target_value


def preprocess_data(sample):
    """
    Preprocess function for the 'modular_arithmetic' task using PyTorch.
    """
    input_sequence, target_value = sample

    # Convert input_sequence to a tensor
    input_tensor = torch.tensor(input_sequence, dtype=torch.long)

    # Create a target tensor of the same shape as the input with only the last value set
    target_tensor = torch.zeros_like(input_tensor, dtype=torch.long)
    target_tensor[-1] = target_value

    # Create a mask where only the last position is True
    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
