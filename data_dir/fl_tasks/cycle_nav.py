import torch

max_position = 5
vocab_size = max_position + 4


def generate_sample(min_length, max_length, seed=None):
    """
    Generates a single sample for the Cycle Nav task using PyTorch, with an optional fixed seed.
    """
    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate sequence length
    length = torch.randint(min_length, max_length + 1, (1,)).item()

    # Generate random movements: 0 = "STAY", 1 = "+1", 2 = "-1"
    movements = torch.randint(0, 3, (length,))
    movement_to_action = ["STAY", "+1", "-1"]

    # Calculate final position using PyTorch operations
    position = (movements == 1).sum() - (movements == 2).sum()
    final_position = 4 + position % max_position

    # Convert movements back to actions for compatibility
    movements = [movement_to_action[m] for m in movements.tolist()]

    return movements, final_position.item()


def preprocess_data(sample):
    """
    Preprocess function for the 'cycle_nav' task using PyTorch.
    """
    movements, final_position = sample

    movement_to_index = {"STAY": 1, "+1": 2, "-1": 3}

    # Convert movements to tensor indices
    input_tensor = torch.tensor(
        [movement_to_index[move] for move in movements], dtype=torch.long
    )

    # Create target tensor
    target_tensor = torch.zeros(input_tensor.shape[0], dtype=torch.long)
    target_tensor[-1] = final_position

    # Create mask tensor
    mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
    mask[-1] = True

    return input_tensor, target_tensor, mask
