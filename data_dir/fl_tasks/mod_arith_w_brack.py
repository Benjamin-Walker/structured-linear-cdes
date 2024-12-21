import torch

vocab_size = 12


def generate_sample(min_length, max_length, generator):
    """
    Generates a single sample for the Modular Arithmetic task with brackets.

    Args:
        min_length: Minimum length of the expression.
        max_length: Maximum length of the expression.
        generator: Random number generator.

    Returns:
        A tuple (input_sequence, target_value), where:
        - input_sequence is a list of integers representing the expression.
        - target_value is the computed result modulo the modulus.
    """

    modulus = vocab_size - 7

    def gen_terminal():
        """Generates a random terminal value."""
        value = generator.randint(0, modulus - 1)
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

        # Split length into left and right parts.
        left_length = generator.randint(1, length - 4)
        right_length = length - (left_length + 3)
        assert left_length >= 1 and right_length >= 1
        left_str, left_val = generate_expression(left_length)
        right_str, right_val = generate_expression(right_length)

        # Sample an operator.
        op = generator.randint(0, 3)
        if op == 0:  # Addition
            return f"({left_str}+{right_str})", (left_val + right_val) % modulus
        elif op == 1:  # Subtraction
            return f"({left_str}-{right_str})", (left_val - right_val) % modulus
        else:  # Multiplication
            return f"({left_str}*{right_str})", (left_val * right_val) % modulus

    # Generate an expression of random length between min_length and max_length.
    length = generator.randint(min_length, max_length)
    expression_str, result = generate_expression(length)

    # Convert expression to numerical representation.
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
    result = vocab[str(result)]
    return input_sequence, result


def preprocess_data(sample):
    """
    Preprocess function for the 'modular_arithmetic' task.

    Args:
        sample: A tuple (input_sequence, target_value).

    Returns:
        A tuple (input_tensor, target_tensor, mask) for PyTorch processing.
    """
    input_sequence, target_value = sample
    input_tensor = torch.tensor(input_sequence, dtype=torch.long)

    # Create a target tensor of the same shape as the input with only the last value set.
    target_tensor = torch.zeros_like(input_tensor, dtype=torch.long)
    target_tensor[-1] = target_value

    # Create a mask where only the last position is True.
    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
