import torch

max_position = 5
vocab_size = max_position + 4


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Cycle Nav task."""

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = generator.randint(min_length, max_length)
    movements = [generator.choice(["+1", "-1", "STAY"]) for _ in range(length)]

    position = 0
    for move in movements:
        if move == "+1":
            position += 1
        elif move == "-1":
            position -= 1

    final_position = 4 + position % max_position

    return movements, final_position


def preprocess_data(sample):
    """Preprocess function for the 'cycle_nav' task."""
    movements, final_position = sample

    movement_to_index = {"STAY": 1, "+1": 2, "-1": 3}
    input_tensor = torch.tensor(
        [movement_to_index[move] for move in movements], dtype=torch.long
    )

    target_tensor = torch.zeros(input_tensor.shape[0], dtype=torch.long)
    target_tensor[-1] = final_position

    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
