import torch

vocab_size = 11


def generate_sample(min_length, max_length, generator):
    """Generates a single sample for the Bucket Sort task."""
    length = generator.randint(min_length, max_length)
    data = [generator.randint(1, vocab_size - 2) for _ in range(length // 2)]
    target = sorted(data)
    data += [10]
    return data, target


def preprocess_data(sample):
    """Preprocess function for the 'bucket_sort' task."""
    data, target = sample
    input_tensor = torch.tensor(data, dtype=torch.long)
    target_tensor = torch.zeros(2 * (input_tensor.shape[0] - 1), dtype=torch.long)
    target_tensor[input_tensor.shape[0] - 1 :] = torch.tensor(target, dtype=torch.long)
    mask = torch.zeros(2 * (input_tensor.shape[0] - 1), dtype=torch.bool)
    mask[input_tensor.shape[0] - 1 :] = True
    return input_tensor, target_tensor, mask
