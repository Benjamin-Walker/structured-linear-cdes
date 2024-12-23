import torch

vocab_size = 11
num_elements = (vocab_size - 3) // 2  # Number of unique stack elements


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Stack Manipulation task."""

    length = generator.randint(min_length, max_length)
    if length % 2 == 1:
        length += 1
    initial_stack = [generator.randint(1, num_elements) for _ in range(length // 4)]

    final_stack = initial_stack.copy()
    operations = []
    for _ in range(length // 4):
        if (
            generator.random() < 0.5 or not final_stack
        ):  # 50% chance or if stack is empty
            operation = generator.randint(1, num_elements)  # Push operation (PSx)
            operations.append(f"PS{operation}")
            final_stack.append(operation)
        else:
            operations.append("POP")
            if final_stack:
                final_stack.pop()

    sequence = (
        [f"ST{el}" for el in initial_stack] + operations + [10]
    )  # Using [ACT] token as 10
    target_sequence = [0] * (len(sequence) - 1) + [f"ST{el}" for el in final_stack]

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'stack_manipulation' task."""
    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(
        [
            int(token[2:])
            if isinstance(token, str) and token.startswith("ST")
            else 9
            if token == "POP"
            else (
                int(token[2:]) + num_elements
                if isinstance(token, str) and token.startswith("PS")
                else token
            )
            for token in input_sequence
        ],
        dtype=torch.long,
    )

    target_tensor = torch.tensor(
        [int(token[2:]) if isinstance(token, str) else 0 for token in target_sequence],
        dtype=torch.long,
    )

    num_nonzero = (target_tensor != 0).sum().item()

    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[-num_nonzero:] = True

    return input_tensor, target_tensor, mask
