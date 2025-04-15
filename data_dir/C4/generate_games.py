import random

import numpy as np
import torch


def generate_random_connect4_game():
    """
    Generates a single random Connect Four game (until the board is full),
    returning:
      - board_states: list of (6,7) numpy arrays after each move
      - move_sequence: list of column indices used at each move

    Board encoding:
      0 = empty
      1 = Red
      2 = Yellow
    """
    # 6 rows x 7 columns, initialized to 0 (empty)
    board = np.zeros((6, 7), dtype=np.int8)

    # next_free_row[c] = row index where the next piece in column c will drop
    next_free_row = [5, 5, 5, 5, 5, 5, 5]

    # Initially, all 7 columns are available
    available_columns = list(range(7))

    board_states = []
    move_sequence = []

    # Player 1 = Red, Player 2 = Yellow; Red starts
    current_player = 1

    while available_columns:
        col = random.choice(available_columns)
        row = next_free_row[col]
        board[row, col] = current_player
        next_free_row[col] -= 1

        # If this column is now full, remove it from the list
        if next_free_row[col] < 0:
            available_columns.remove(col)

        move_sequence.append(col)
        board_states.append(board.copy())

        # Switch player
        current_player = 3 - current_player

    return board_states, move_sequence


def generate_sample_c4(seed=None):
    """
    Generates a single Connect Four "sample".
    Returns:
      sequence: A list of board states (each state will be flattened later).
      target:   The corresponding list of moves chosen for each state.
    """
    # If a seed is specified, set the random seed
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    board_states, move_sequence = generate_random_connect4_game()

    # Each board state is a 6x7 np array; we'll keep them as lists of length 42
    sequence = move_sequence
    target = [state.flatten().tolist() for state in board_states]

    return sequence, target


def preprocess_data_c4(sample):
    """
    Preprocess function for Connect Four data.
    Returns: (input_tensor, target_tensor, mask)
      - input_tensor: shape (L, 42), where L is the number of moves
      - target_tensor: shape (L,), columns are offset by +1 so 1..7 = valid columns, 0 = padding
      - mask: shape (L,), all True so we predict each move
    """
    moves, board = sample  # sequence: list of length L
    L = len(moves)

    # For the moves (columns 0..6), we shift them by +1 so 1..7 become “real” classes; 0 will be used for padding
    input_tensor = torch.tensor([m + 1 for m in moves], dtype=torch.long)  # shape: (L,)

    # Turn each flattened board state into a long tensor
    target_tensor = torch.tensor(board, dtype=torch.long)

    # We want to predict each board, so mask is all True
    mask = torch.ones(L, dtype=torch.bool)

    return input_tensor, target_tensor, mask
