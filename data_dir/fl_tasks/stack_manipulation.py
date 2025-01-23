import torch

num_elements = 2  # Number of unique stack elements
vocab_size = num_elements * 2 + 4  # padding, states, push actions, pop, empty list, act


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Stack Manipulation task."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = torch.randint(min_length, max_length + 1, (1,)).item()

    # length gives the total length of target_sequence
    if length % 2 == 1:
        length += 1

    # Initialize the stack content and the actions
    initial_stack_length = torch.randint(1, length // 2 + 1, (1,)).item()
    initial_stack = [
        torch.randint(1, num_elements + 1, (1,)).item()
        for _ in range(initial_stack_length)
    ]
    final_stack = initial_stack.copy()

    actions = [
        torch.randint(0, num_elements + 1, (1,)).item()
        for _ in range(length // 2 - initial_stack_length)
    ]

    operations = []

    for action in actions:
        if action == 0:
            if final_stack:
                final_stack.pop()
            operations.append("POP")
        else:
            final_stack.append(action)
            operations.append(f"PS{action}")

    sequence = (
        [f"ST{el}" for el in initial_stack] + operations + [int(vocab_size - 1)]
    )  # Using [ACT] token as the end token
    target_sequence = (
        [0] * (len(sequence) - 1)
        + (["EMPTY"] if not final_stack else [f"ST{el}" for el in final_stack])
        + [0] * (length // 2 - max(1, len(final_stack)))
    )

    return sequence, target_sequence


def preprocess_data(sample):
    """Preprocess function for the 'stack_manipulation' task."""
    input_sequence, target_sequence = sample

    input_tensor = torch.tensor(
        [
            int(token[2:])
            if isinstance(token, str) and token.startswith("ST")
            else int(num_elements * 2 + 1)
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
        [
            num_elements * 2 + 2
            if token == "EMPTY"
            else int(token[2:])
            if isinstance(token, str)
            else 0
            for token in target_sequence
        ],
        dtype=torch.long,
    )

    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    ind = torch.argmax((target_tensor != 0).to(dtype=torch.int), dim=-1)
    mask[ind:] = True

    return input_tensor, target_tensor, mask
