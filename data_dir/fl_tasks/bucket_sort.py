import torch

num_elements = 5
vocab_size = num_elements + 2


def generate_sample(min_length, max_length, seed=None):
    """Generates a single sample for the Bucket Sort task."""

    # Set the seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    if min_length > max_length:
        raise ValueError("min_length must be less than or equal to max_length")

    length = torch.randint(min_length, max_length + 1, (1,)).item()
    if length % 2 == 1:
        length += 1

    data = [torch.randint(1, num_elements + 1, (1,)).item() for _ in range(length // 2)]
    target = [0] * (length // 2) + sorted(data)
    data.append(vocab_size - 1)
    return data, target


def preprocess_data(sample):
    """Preprocess function for the 'bucket_sort' task."""

    data, target = sample
    input_tensor = torch.tensor(data, dtype=torch.long)
    target_tensor = torch.tensor(target, dtype=torch.long)
    mask = torch.zeros(target_tensor.shape, dtype=torch.bool)
    mask[len(data) - 1 :] = True
    return input_tensor, target_tensor, mask
