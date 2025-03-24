import random

import torch

from data_dir.fl_tasks.binary_addition import (
    generate_sample as generate_sample_binary_addition,
)
from data_dir.fl_tasks.binary_addition import (
    preprocess_data as preprocess_data_binary_addition,
)
from data_dir.fl_tasks.binary_multiplication import (
    generate_sample as generate_sample_binary_multiplication,
)
from data_dir.fl_tasks.binary_multiplication import (
    preprocess_data as preprocess_data_binary_multiplication,
)
from data_dir.fl_tasks.bucket_sort import generate_sample as generate_sample_bucket_sort
from data_dir.fl_tasks.bucket_sort import preprocess_data as preprocess_data_bucket_sort
from data_dir.fl_tasks.compute_sqrt import (
    generate_sample as generate_sample_compute_sqrt,
)
from data_dir.fl_tasks.compute_sqrt import (
    preprocess_data as preprocess_data_compute_sqrt,
)
from data_dir.fl_tasks.cycle_nav import generate_sample as generate_sample_cycle_nav
from data_dir.fl_tasks.cycle_nav import preprocess_data as preprocess_data_cycle_nav
from data_dir.fl_tasks.duplicate_string import (
    generate_sample as generate_sample_duplicate_string,
)
from data_dir.fl_tasks.duplicate_string import (
    preprocess_data as preprocess_data_duplicate_string,
)
from data_dir.fl_tasks.even_pairs import generate_sample as generate_sample_even_pairs
from data_dir.fl_tasks.even_pairs import preprocess_data as preprocess_data_even_pairs
from data_dir.fl_tasks.majority import generate_sample as generate_sample_majority
from data_dir.fl_tasks.majority import preprocess_data as preprocess_data_majority
from data_dir.fl_tasks.majority_count import (
    generate_sample as generate_sample_majority_count,
)
from data_dir.fl_tasks.majority_count import (
    preprocess_data as preprocess_data_majority_count,
)
from data_dir.fl_tasks.missing_duplicate import (
    generate_sample as generate_sample_missing_duplicate,
)
from data_dir.fl_tasks.missing_duplicate import (
    preprocess_data as preprocess_data_missing_duplicate,
)
from data_dir.fl_tasks.mod_arith_no_brack import (
    generate_sample as generate_sample_mod_arith_no_brack,
)
from data_dir.fl_tasks.mod_arith_no_brack import (
    preprocess_data as preprocess_data_mod_arith_no_brack,
)
from data_dir.fl_tasks.mod_arith_w_brack import (
    generate_sample as generate_sample_mod_arith_w_brack,
)
from data_dir.fl_tasks.mod_arith_w_brack import (
    preprocess_data as preprocess_data_mod_arith_w_brack,
)
from data_dir.fl_tasks.odds_first import generate_sample as generate_sample_odds_first
from data_dir.fl_tasks.odds_first import preprocess_data as preprocess_data_odds_first
from data_dir.fl_tasks.parity import generate_sample as generate_sample_parity
from data_dir.fl_tasks.parity import preprocess_data as preprocess_data_parity
from data_dir.fl_tasks.reverse_string import (
    generate_sample as generate_sample_reverse_string,
)
from data_dir.fl_tasks.reverse_string import (
    preprocess_data as preprocess_data_reverse_string,
)
from data_dir.fl_tasks.set import generate_sample as generate_sample_set
from data_dir.fl_tasks.set import preprocess_data as preprocess_data_set
from data_dir.fl_tasks.solve_equation import (
    generate_sample as generate_sample_solve_equation,
)
from data_dir.fl_tasks.solve_equation import (
    preprocess_data as preprocess_data_solve_equation,
)
from data_dir.fl_tasks.stack_manipulation import (
    generate_sample as generate_sample_stack_manipulation,
)
from data_dir.fl_tasks.stack_manipulation import (
    preprocess_data as preprocess_data_stack_manipulation,
)


def test_even_pairs_generate():
    for i in range(1000):
        sample = generate_sample_even_pairs(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], str)
        assert isinstance(sample[1], str)
        assert 3 <= len(sample[0]) <= 10
        assert sample[1] in ["a", "b"]
        assert sample[1] == ("a" if sample[0][0] == sample[0][-1] else "b")

    sample = generate_sample_even_pairs(3, 3, 0)
    assert len(sample[0]) == 3

    sample = generate_sample_even_pairs(10, 10, 0)
    assert len(sample[0]) == 10


def test_even_pairs_preprocess():
    test_cases = [
        (("ab", "b"), [1, 2], [0, 2], [0, 1]),
        (("aa", "a"), [1, 1], [0, 1], [0, 1]),
        (("bab", "b"), [2, 1, 2], [0, 0, 2], [0, 0, 1]),
    ]

    for sample, expected_input, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_even_pairs(sample)
        assert input_tensor.shape == (len(expected_input),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(expected_input)).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


modular_arithmetic_dict = {
    0: "[PAD]",
    1: "+",
    2: "-",
    3: "*",
    4: "=",
    5: "0",
    6: "1",
    7: "2",
    8: "3",
    9: "4",
}


def test_mod_arith_no_brack_generate():
    for i in range(1000):
        sample = generate_sample_mod_arith_no_brack(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], int)
        # Length should be even, and the last token should be '='
        assert 4 <= len(sample[0]) <= 10
        assert sample[0][-1] == 4
        assert sample[1] in range(5, 10)
        question = "".join([modular_arithmetic_dict[i] for i in sample[0]][:-1])
        assert sample[1] == eval(question) % 5 + 5

    sample = generate_sample_mod_arith_no_brack(3, 3, 0)
    assert len(sample[0]) == 4

    sample = generate_sample_mod_arith_no_brack(10, 10, 0)
    assert len(sample[0]) == 10


def test_mod_arith_no_brack_preprocess():
    sample = ([6, 1, 7, 3, 6, 2, 7, 4], 9)
    input_tensor, target_tensor, mask = preprocess_data_mod_arith_no_brack(sample)
    assert input_tensor.shape == (len(sample[0]),)
    assert target_tensor.shape == (len(sample[0]),)
    assert (input_tensor == torch.tensor([6, 1, 7, 3, 6, 2, 7, 4])).all()
    assert (
        target_tensor == torch.tensor([0] * (len(sample[0]) - 1) + [sample[1]])
    ).all()
    assert (mask == torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool)).all()


def test_majority_generate():
    """Tests the generate_sample_majority function."""
    for i in range(1000):
        sample = generate_sample_majority(3, 10, random.Random(i))

        # Check the structure of the output
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], int)

        assert 3 <= len(sample[0]) <= 10

        tokens = sample[0]
        majority_token = sample[1]
        token_counts = {}

        # Count the occurrences using a dictionary
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

        # Determine the maximum count
        max_count = max(token_counts.values())

        # Find all tokens with the maximum count
        candidates = [
            token for token, count in token_counts.items() if count == max_count
        ]

        # Find the first occurrence in the original list among the candidates
        expected_majority_token = None
        for token in tokens:
            if token in candidates:
                expected_majority_token = token
                break

        assert majority_token == expected_majority_token


def test_majority_preprocess():
    """Tests the preprocess_data_majority function."""
    # Sample inputs and expected outputs
    test_cases = [
        ([3, 3, 2, 4, 4, 4], 4, [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 1]),
        ([1, 1, 1, 2, 2, 2, 2], 2, [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 1]),
        ([5, 5, 5, 3, 3], 5, [0, 0, 0, 0, 5], [0, 0, 0, 0, 1]),
        ([9], 9, [9], [1]),
        (
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            1,
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ),
    ]

    for tokens, expected_majority_token, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_majority(
            (tokens, expected_majority_token)
        )

        # Verify the shape and content of the input tensor
        assert input_tensor.shape == (len(tokens),)
        assert (input_tensor == torch.tensor(tokens, dtype=torch.long)).all()

        # Verify the shape and content of the target tensor
        assert target_tensor.shape == (len(expected_target),)
        assert (target_tensor == torch.tensor(expected_target, dtype=torch.long)).all()

        # Verify the shape and content of the mask tensor
        assert mask.shape == (len(tokens),)
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_generate_sample_majority_count():
    """Tests the generate_sample_majority_count function."""
    for i in range(1000):
        sample = generate_sample_majority_count(3, 10, random.Random(i))

        # Check the structure of the output
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], int)

        # Check the length of the token list
        assert 3 <= len(sample[0]) <= 10

        tokens = sample[0]
        majority_count = sample[1]

        token_counts = {}

        # Count the occurrences using a dictionary
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

        # Determine the maximum count
        max_count = max(token_counts.values())

        # Verify that the returned majority count matches the expected maximum count
        assert majority_count == max_count


def test_preprocess_data_majority_count():
    """Tests the preprocess_data_majority_count function."""
    # Sample inputs and expected outputs
    test_cases = [
        ([3, 3, 2, 4, 4, 4], 4, [0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 1]),
        ([1, 1, 1, 2, 2, 2, 2], 2, [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 1]),
        ([5, 5, 5, 3, 3], 5, [0, 0, 0, 0, 5], [0, 0, 0, 0, 1]),
        ([9], 9, [9], [1]),
        (
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            1,
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ),
    ]

    for tokens, expected_majority_count, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_majority_count(
            (tokens, expected_majority_count)
        )

        # Verify the shape and content of the input tensor
        assert input_tensor.shape == (len(tokens),)
        assert (input_tensor == torch.tensor(tokens, dtype=torch.long)).all()

        # Verify the shape and content of the target tensor
        assert target_tensor.shape == (len(expected_target),)
        assert (target_tensor == torch.tensor(expected_target, dtype=torch.long)).all()

        # Verify the shape and content of the mask tensor
        assert mask.shape == (len(tokens),)
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_bucket_sort_generate():
    vocab_size = 7

    for i in range(1000):
        sample = generate_sample_bucket_sort(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        # Token should be half of the length, and the last token should be [ACT]
        assert 3 // 2 <= len(sample[0][:-1]) <= 10 // 2
        assert sample[0][-1] == vocab_size - 1
        assert sample[1][len(sample[0]) - 1 :] == sorted(sample[0][:-1])

    sample = generate_sample_bucket_sort(4, 4, 0)
    assert len(sample[0][:-1]) == 4 // 2

    sample = generate_sample_bucket_sort(10, 10, 0)
    assert len(sample[0][:-1]) == 10 // 2


def test_bucket_sort_preprocess():
    vocab_size = 7
    test_cases = [
        (
            [4, 2, 3, 1, 5, vocab_size - 1],
            [0] * 5 + [1, 2, 3, 4, 5],
            [0] * 5 + [1, 2, 3, 4, 5],
            [0] * 5 + [1] * 5,
        ),
        (
            [2, 2, 3, 4, 5, 1, 1, 3, 4, vocab_size - 1],
            [0] * 9 + [1, 1, 2, 2, 3, 3, 4, 4, 5],
            [0] * 9 + [1, 1, 2, 2, 3, 3, 4, 4, 5],
            [0] * 9 + [1] * 9,
        ),
        ([1, vocab_size - 1], [0, 1], [0, 1], [0, 1]),
    ]

    for tokens, answer, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_bucket_sort(
            (tokens, answer)
        )
        assert input_tensor.shape == (len(tokens),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(tokens, dtype=torch.long)).all()
        assert (target_tensor == torch.tensor(expected_target, dtype=torch.long)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_cycle_nav_generate():
    for i in range(1000):
        sample = generate_sample_cycle_nav(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], int)
        assert 3 <= len(sample[0]) <= 10
        assert 4 <= sample[1] <= 8

    sample = generate_sample_cycle_nav(3, 3, 0)
    assert len(sample[0]) == 3

    sample = generate_sample_cycle_nav(10, 10, 0)
    assert len(sample[0]) == 10


def test_cycle_nav_preprocess():
    test_cases = [
        ([["+1", "-1", "STAY"], 4], [2, 3, 1], [0, 0, 4], [0, 0, 1]),
        ([["+1", "-1", "STAY", "+1"], 5], [2, 3, 1, 2], [0, 0, 0, 5], [0, 0, 0, 1]),
        ([["STAY", "-1"], 8], [1, 3], [0, 8], [0, 1]),
    ]

    for sample, expected_input, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_cycle_nav(sample)
        assert input_tensor.shape == (len(expected_input),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(expected_input)).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_missing_duplicate_generate():
    vocab_size = 4

    for i in range(1000):
        sample = generate_sample_missing_duplicate(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], int)
        assert 4 <= len(sample[0]) <= 10
        assert sample[1] in range(1, 10)
        assert vocab_size - 1 in sample[0]

    sample = generate_sample_missing_duplicate(3, 3, 0)
    assert len(sample[0]) == 4

    sample = generate_sample_missing_duplicate(10, 10, 0)
    assert len(sample[0]) == 10


def test_missing_duplicate_preprocess():
    vocab_size = 4
    test_cases = [
        (
            [1, 2, 1, 2, 1, 2, 1, vocab_size - 1],
            2,
            [0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ),
        ([1, 2, 1, 1, vocab_size - 1, 1], 2, [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 1, 0]),
        ([1, vocab_size - 1], 1, [0, 1], [0, 1]),
    ]

    for sample, expected_missing_token, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_missing_duplicate(
            (sample, expected_missing_token)
        )
        assert input_tensor.shape == (len(sample),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(sample)).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_odds_first_generate():
    for i in range(1000):
        sample = generate_sample_odds_first(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        assert 3 <= len(sample[0]) <= 6
        assert 4 <= len(sample[1]) <= 10
        length = len(sample[0]) - 1
        assert sample[0][:-1][::2] == sample[1][length : length + (length + 1) // 2]
        assert sample[0][:-1][1::2] == sample[1][length + (length + 1) // 2 :]

    sample = generate_sample_odds_first(3, 3, 0)
    assert len(sample[0]) == 3
    assert len(sample[1]) == 4

    sample = generate_sample_odds_first(10, 10, 0)
    assert len(sample[0]) == 6
    assert len(sample[1]) == 10


def test_odds_first_preprocess():
    vocab_size = 4

    test_cases = [
        (
            ([1, 2, 1, 2, 1, vocab_size - 1], [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]),
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ),
        (
            ([2, 2, 1, vocab_size - 1], [0, 0, 0, 2, 1, 2]),
            [0, 0, 0, 2, 1, 2],
            [0, 0, 0, 1, 1, 1],
        ),
        (([1, vocab_size - 1], [0, 1]), [0, 1], [0, 1]),
    ]

    for sample, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_odds_first(sample)
        assert input_tensor.shape == (len(sample[0]),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(sample[0])).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_parity_generate():
    for i in range(1000):
        sample = generate_sample_parity(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], str)
        assert 3 <= len(sample[0]) <= 10
        assert sample[1] in ["a", "b"]
        assert sample[1] == ("a" if sample[0].count("b") % 2 == 0 else "b")

    sample = generate_sample_parity(3, 3, 0)
    assert len(sample[0]) == 3

    sample = generate_sample_parity(10, 10, 0)
    assert len(sample[0]) == 10


def test_parity_preprocess():
    test_cases = [
        (
            (["a", "b", "a", "b", "b"], "b"),
            [1, 2, 1, 2, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 1],
        ),
        (
            (["a", "a", "b", "b", "b"], "b"),
            [1, 1, 2, 2, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 1],
        ),
        (
            (["b", "a", "a", "b", "a"], "a"),
            [2, 1, 1, 2, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ),
    ]

    for sample, expected_input, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_parity(sample)
        assert input_tensor.shape == (len(expected_input),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(expected_input)).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_duplicate_string_generate():
    for i in range(1000):
        sample = generate_sample_duplicate_string(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        assert 2 <= len(sample[0]) <= 6
        assert sample[1] == sample[0][:-1] + sample[0][:-1]

    sample = generate_sample_duplicate_string(3, 3, 0)
    assert len(sample[0]) == 3
    assert len(sample[1]) == 4

    sample = generate_sample_duplicate_string(10, 10, 0)
    assert len(sample[0]) == 6
    assert len(sample[1]) == 10


def test_duplicate_string_preprocess():
    vocab_size = 4

    test_cases = [
        (
            ([1, 2, 1, vocab_size - 1], [1, 2, 1, 1, 2, 1]),
            [1, 2, 1, 1, 2, 1],
            [0, 0, 0, 1, 1, 1],
        ),
        (([1, 1, vocab_size - 1], [1, 1, 1, 1]), [1, 1, 1, 1], [0, 0, 1, 1]),
        (([2, vocab_size - 1], [2, 2]), [2, 2], [0, 1]),
    ]

    for sample, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_duplicate_string(sample)
        assert input_tensor.shape == (len(sample[0]),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(sample[0])).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_reverse_string_generate():
    for i in range(1000):
        sample = generate_sample_reverse_string(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        assert 2 <= len(sample[0]) <= 6
        assert sample[1] == sample[0][:-1] + sample[0][:-1][::-1]

    sample = generate_sample_reverse_string(3, 3, 0)
    assert len(sample[0]) == 3
    assert len(sample[1]) == 4

    sample = generate_sample_reverse_string(10, 10, 0)
    assert len(sample[0]) == 6
    assert len(sample[1]) == 10


def test_reverse_string_preprocess():
    test_cases = [
        (
            ([1, 2, 3, 11], [1, 2, 3, 1, 2, 1]),
            [1, 2, 3, 1, 2, 1],
            [0, 0, 0, 1, 1, 1],
        ),
        (([3, 1, 11], [3, 1, 3, 1]), [3, 1, 3, 1], [0, 0, 1, 1]),
        (([7, 11], [7, 7]), [7, 7], [0, 1]),
    ]

    for sample, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_reverse_string(sample)
        assert input_tensor.shape == (len(sample[0]),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(sample[0])).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_stack_manipulation_generate():
    num_elements = 2
    for i in range(1000):
        sample = generate_sample_stack_manipulation(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        assert len(sample[0]) == len(sample[1]) / 2 + 1
        assert 2 <= len(sample[0]) <= 6
        assert 4 <= len(sample[1]) <= 10
        stack = []
        for token in sample[0]:
            if isinstance(token, int):
                break
            if token.startswith("ST"):
                stack.append(int(token[2:]))
            elif token == "POP" and stack:
                stack.pop()
            elif token.startswith("PS"):
                stack.append(int(token[2:]))

        stack = (
            [0] * (len(sample[0]) - 1)
            + ([num_elements * 2 + 2] if not stack else stack)
            + [0] * (len(sample[0]) - 1 - max(1, len(stack)))
        )

        pred_stack = [
            num_elements * 2 + 2
            if token == "EMPTY"
            else int(token[2:])
            if isinstance(token, str)
            else 0
            for token in sample[1]
        ]
        assert stack == pred_stack

    sample = generate_sample_stack_manipulation(3, 3, 0)
    assert len(sample[0]) == 3
    assert len(sample[1]) == 4

    sample = generate_sample_stack_manipulation(10, 10, 0)
    assert len(sample[0]) == 6
    assert len(sample[1]) == 10


def test_stack_manipulation_preprocess():
    num_elements = 2
    act_token = num_elements * 2 + 3
    test_cases = [
        (
            (
                ["ST1", "ST2", "PS2", "POP", "PS1", "POP", "POP", "POP", act_token],
                [0, 0, 0, 0, 0, 0, 0, 0, "EMPTY", 0, 0, 0, 0, 0, 0, 0],
            ),
            [1, 2, 4, 5, 3, 5, 5, 5, act_token],
            [0, 0, 0, 0, 0, 0, 0, 0, num_elements * 2 + 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        (
            (
                ["ST1", "PS2", "PS1", "POP", act_token],
                [0, 0, 0, 0, "ST1", "ST2", 0, 0],
            ),
            [1, 4, 3, 5, act_token],
            [0, 0, 0, 0, 1, 2, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ),
        (
            (["ST2", "PS1", act_token], [0, 0, "ST2", "ST1"]),
            [2, 3, act_token],
            [0, 0, 2, 1],
            [0, 0, 1, 1],
        ),
    ]

    for sample, expected_input, expected_output, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_stack_manipulation(sample)
        assert input_tensor.shape == (len(sample[0]),)
        assert target_tensor.shape == (len(sample[1]),)
        assert (input_tensor == torch.tensor(expected_input)).all()
        assert (target_tensor == torch.tensor(expected_output)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_set_generate():
    from collections import OrderedDict

    for i in range(1000):
        sample = generate_sample_set(3, 10, random.Random(i))
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        assert 3 <= len(sample[0]) <= 6
        num_non_zero = sum([x != 0 for x in sample[1]])
        answer = list(OrderedDict.fromkeys(sample[0][:-1]))
        assert sample[1][-num_non_zero:] == answer

    sample = generate_sample_set(3, 3, random.Random(0))
    assert len(sample[0]) == 3
    assert len(sample[1]) == 4

    sample = generate_sample_set(10, 10, random.Random(0))
    assert len(sample[0]) == 6
    assert len(sample[1]) == 10


def test_set_preprocess():
    test_cases = [
        (
            ([1, 2, 3, 4, 10], [0, 0, 0, 0, 1, 2, 3, 4]),
            [0, 0, 0, 0, 1, 2, 3, 4],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ),
        (
            ([1, 1, 2, 2, 10], [0, 0, 0, 0, 1, 2]),
            [0, 0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1, 1],
        ),
        (([7, 10], [0, 7]), [0, 7], [0, 1]),
    ]

    for sample, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_set(sample)
        assert input_tensor.shape == (len(sample[0]),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(sample[0])).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_generate_sample_mod_arith_with_brackets():
    modulus = 5
    vocab = {
        "+": modulus + 1,
        "-": modulus + 2,
        "*": modulus + 3,
        "(": modulus + 4,
        ")": modulus + 5,
    }
    for i in range(modulus):
        vocab[str(i)] = i + 1

    inv_vocab = {v: k for k, v in vocab.items()}  # Invert the vocab

    for i in range(1000):
        sample = generate_sample_mod_arith_w_brack(5, 20, i)
        input_sequence, result_token = sample

        assert isinstance(input_sequence, list)
        assert all(isinstance(token, int) for token in input_sequence[:-1])

        assert isinstance(result_token, int)

        expression_str = "".join(inv_vocab[token] for token in input_sequence[:-1])

        allowed_chars = set("0123456789+-*()")
        assert set(expression_str).issubset(allowed_chars)

        try:
            eval_result = eval(expression_str, {"__builtins__": None}, {})
            eval_result = eval_result % modulus
        except Exception as e:
            assert False, f"Failed to evaluate expression '{expression_str}': {e}"

        expected_result_token = vocab[str(eval_result)]
        assert (
            result_token == expected_result_token
        ), f"Expected result token {expected_result_token}, got {result_token}."


def test_preprocess_data_mod_arith_with_brackets():
    test_cases = [
        # Expression: (-2+3) mod 5 = 1
        (
            [9, 7, 3, 6, 4, 10],
            2,  # Input sequence and target token
            [9, 7, 3, 6, 4, 10],  # Expected input tensor
            [0, 0, 0, 0, 0, 2],  # Expected target tensor
            [0, 0, 0, 0, 0, 1],  # Expected mask
        ),
        # Expression: (3*(1+4)) mod 5 = (3*0) mod 5 = 0
        (
            [9, 4, 8, 9, 2, 6, 5, 10, 10],
            1,
            [9, 4, 8, 9, 2, 6, 5, 10, 10],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ),
        # Expression: -(-3) mod 5 = 3
        (
            [7, 9, 7, 4, 10],
            5,
            [7, 9, 7, 4, 10],
            [0, 0, 0, 0, 5],
            [0, 0, 0, 0, 1],
        ),
    ]

    for (
        input_sequence,
        target_value,
        expected_input,
        expected_target,
        expected_mask,
    ) in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_mod_arith_w_brack(
            (input_sequence, target_value)
        )

        assert (
            input_tensor == torch.tensor(expected_input, dtype=torch.long)
        ).all(), f"Expected input tensor {expected_input}, got {input_tensor.tolist()}."

        assert (
            target_tensor == torch.tensor(expected_target, dtype=torch.long)
        ).all(), (
            f"Expected target tensor {expected_target}, got {target_tensor.tolist()}."
        )

        expected_mask_tensor = torch.tensor(expected_mask, dtype=torch.bool)
        assert (
            mask == expected_mask_tensor
        ).all(), f"Expected mask {expected_mask}, got {mask.tolist()}."


def test_generate_sample_solve_equation():
    modulus = 5

    vocab = {
        1: "0",
        2: "1",
        3: "2",
        4: "3",
        5: "4",
        6: "+",
        7: "-",
        8: "*",
        9: "(",
        10: ")",
        11: "=",
        12: "x",
        13: "[ACT]",
    }

    for i in range(1000):
        sequence, eqn_solution = generate_sample_solve_equation(5, 20, i)

        assert isinstance(sequence, list), "Sequence should be a list."
        assert all(
            isinstance(token, int) for token in sequence
        ), "All tokens should be integers."

        assert isinstance(eqn_solution, int), "Equation solution should be an integer."

        # Extract the expression (excluding the last three tokens: '=', result, '[ACT]')
        expression_tokens = sequence[:-3]
        expression_str = "".join(vocab[token] for token in expression_tokens)

        # Replace 'x' with the actual solution in the expression
        expression_with_solution = expression_str.replace("x", str(eqn_solution))

        try:
            eval_result = eval(expression_with_solution, {"__builtins__": None}, {})
            eval_result = eval_result % modulus
        except Exception as e:
            assert (
                False
            ), f"Failed to evaluate expression '{expression_with_solution}': {e}"

        provided_solution_token = sequence[-2]
        provided_solution = int(vocab[provided_solution_token])

        assert (
            eval_result == provided_solution
        ), f"Expected solution {eval_result}, got {provided_solution}."


def test_preprocess_data_solve_equation():
    test_cases = [
        # Expression: (2 + x) = 1 mod 5, where x = 4
        (
            [2, 5, 12, 10, 1, 13],
            4,  # Input sequence and actual value of x
            [2, 5, 12, 10, 1, 13],  # Expected input tensor
            [0, 0, 0, 0, 0, 4],  # Expected target tensor
            [0, 0, 0, 0, 0, 1],  # Expected mask
        ),
        # Expression: (x * 3) = 2 mod 5, where x = 4
        (
            [8, 12, 7, 3, 9, 10, 2, 13],
            4,
            [8, 12, 7, 3, 9, 10, 2, 13],
            [0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ),
        # Expression: -x = 3 mod 5, where x = 2
        (
            [6, 12, 10, 3, 13],
            2,
            [6, 12, 10, 3, 13],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 1],
        ),
    ]

    for (
        sequence,
        eqn_solution,
        expected_input,
        expected_target,
        expected_mask,
    ) in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_solve_equation(
            (sequence, eqn_solution)
        )

        assert (
            input_tensor == torch.tensor(expected_input, dtype=torch.long)
        ).all(), f"Expected input tensor {expected_input}, got {input_tensor.tolist()}."

        assert (
            target_tensor == torch.tensor(expected_target, dtype=torch.long)
        ).all(), (
            f"Expected target tensor {expected_target}, got {target_tensor.tolist()}."
        )

        expected_mask_tensor = torch.tensor(expected_mask, dtype=torch.bool)
        assert (
            mask == expected_mask_tensor
        ).all(), f"Expected mask {expected_mask.tolist()}, got {mask.tolist()}."


def test_binary_addition_generate():
    vocab_size = 5

    for i in range(1000):
        sample = generate_sample_binary_addition(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        assert 4 <= len(sample[0]) <= 6
        assert vocab_size - 1 in sample[0]
        assert vocab_size - 2 in sample[0]

    sample = generate_sample_binary_addition(5, 5, 0)
    assert len(sample[1]) == 6

    sample = generate_sample_binary_addition(10, 10, 0)
    assert len(sample[1]) == 10


def test_binary_addition_preprocess():
    test_cases = [
        (
            ([2, 2, 3, 1, 1, 2, 4], [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0]),
            [0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ),
        (([2, 3, 2, 4], [0, 0, 0, 1, 2, 0]), [0, 0, 0, 1, 2, 0], [0, 0, 0, 1, 1, 1]),
    ]

    for sample, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_binary_addition(sample)
        assert input_tensor.shape == (len(sample[0]),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(sample[0])).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_binary_multiplication_generate():
    vocab_size = 5

    for i in range(1000):
        sample = generate_sample_binary_multiplication(3, 10, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        assert 4 <= len(sample[0]) <= 6
        assert vocab_size - 1 in sample[0]
        assert vocab_size - 2 in sample[0]

    sample = generate_sample_binary_multiplication(5, 5, 0)
    assert len(sample[1]) == 6

    sample = generate_sample_binary_multiplication(10, 10, 0)
    assert len(sample[1]) == 10


def test_binary_multiplication_preprocess():
    test_cases = [
        (
            ([2, 2, 3, 1, 1, 2, 4], [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0]),
            [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ),
        (([2, 3, 2, 4], [0, 0, 0, 2, 0, 0]), [0, 0, 0, 2, 0, 0], [0, 0, 0, 1, 1, 1]),
    ]

    for sample, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_binary_multiplication(
            sample
        )
        assert input_tensor.shape == (len(sample[0]),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(sample[0])).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()


def test_binary_compute_sqrt_generate():
    vocab_size = 4

    for i in range(50):
        sample = generate_sample_compute_sqrt(5, 40, i)
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert isinstance(sample[0], list)
        assert isinstance(sample[1], list)
        assert 5 <= len(sample[1]) <= 40
        assert vocab_size - 1 in sample[0]

    sample = generate_sample_compute_sqrt(6, 6, 0)
    assert len(sample[1]) == 5

    sample = generate_sample_compute_sqrt(9, 9, 0)
    assert len(sample[1]) == 7


def test_compute_sqrt_preprocess():
    test_cases = [
        (
            ([1, 1, 2, 2, 2, 4], [0, 0, 0, 0, 0, 2, 1, 2]),
            [0, 0, 0, 0, 0, 2, 1, 2],
            [0, 0, 0, 0, 0, 1, 1, 1],
        ),
        (
            ([1, 1, 2, 2, 4], [0, 0, 0, 0, 2, 2, 0]),
            [0, 0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 1, 1, 1],
        ),
    ]

    for sample, expected_target, expected_mask in test_cases:
        input_tensor, target_tensor, mask = preprocess_data_compute_sqrt(sample)
        assert input_tensor.shape == (len(sample[0]),)
        assert target_tensor.shape == (len(expected_target),)
        assert (input_tensor == torch.tensor(sample[0])).all()
        assert (target_tensor == torch.tensor(expected_target)).all()
        assert (mask == torch.tensor(expected_mask, dtype=torch.bool)).all()
