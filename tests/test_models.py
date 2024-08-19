import torch

from models.slcde import A5LinearCDE, Embedding, LinearCDE


# Test case for the Embedding class
def test_embedding_forward():
    num_embeddings = 10
    embedding_dim = 4
    batch_size = 5
    embedding_layer = Embedding(num_embeddings, embedding_dim)

    # Input tensor of indices
    x = torch.randint(0, num_embeddings, (batch_size,))

    # Perform forward pass
    output = embedding_layer(x)

    # Check the output shape
    assert output.shape == (batch_size, embedding_dim)

    # Check if output values match the weights at corresponding indices
    torch.allclose(output, embedding_layer.weights[x])


# Test case for the LinearCDE class
def test_linearcde_forward():
    batch_size = 2
    seq_len = 3
    hidden_dim = 5
    data_dim = 4

    linear_cde = LinearCDE(hidden_dim, data_dim)

    # Input tensor (batch_size, seq_len, data_dim)
    X = torch.randn(batch_size, seq_len, data_dim)

    # Perform forward pass
    output = linear_cde(X)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, hidden_dim)


# Test case for the A5LinearCDE class
def test_a5linearcde_forward():
    batch_size = 2
    seq_len = 3
    hidden_dim = 5
    data_dim = 4
    label_dim = 6

    a5linearcde = A5LinearCDE(hidden_dim, data_dim, label_dim)

    # Input tensor (batch_size, seq_len, data_dim)
    X = torch.randint(0, label_dim, (batch_size, seq_len))

    # Perform forward pass
    output = a5linearcde(X)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, label_dim)


# Check if linearcde gives the depth-1 signature for the right initialisation
def test_increment_linearcdee():
    hidden_dim = 3
    data_dim = 1

    linear_cde = LinearCDE(hidden_dim, data_dim)
    with torch.no_grad():
        linear_cde.init_layer.weight.data.fill_(0)
        linear_cde.vf_B.weight.fill_(0)
        bias_init = torch.zeros(hidden_dim)
        bias_init[0] = 1
        linear_cde.init_layer.bias = torch.nn.Parameter(bias_init)

        vf_A = torch.zeros(hidden_dim * (data_dim + 1), hidden_dim)
        vf_A[2, 0] = 1
        vf_A[5, 0] = 1
        linear_cde.vf_A.weight = torch.nn.Parameter(vf_A)

    X = torch.Tensor([[[0], [1.5], [-1.5]]])

    output = linear_cde(X)

    # Linear CDE with the given initialisation should give the depth-1 signature of the
    # two dimensional path with the first dimension as time from 0 to 1 and the second
    # dimension as cumulative sum of the input
    assert (output == torch.Tensor([[[1, 0, 0], [1, 0.5, 1.5], [1, 1.0, 0.0]]])).all()


# Additional test to check dropout behavior
def test_a5linearcde_dropout():
    batch_size = 2
    seq_len = 3
    hidden_dim = 5
    data_dim = 4
    label_dim = 6

    a5linearcde = A5LinearCDE(hidden_dim, data_dim, label_dim)

    X = torch.randint(0, label_dim, (batch_size, seq_len))

    # Set model to evaluation mode
    a5linearcde.eval()
    output_no_dropout = a5linearcde(X)

    # Set model to training mode
    a5linearcde.train()
    output_with_dropout = a5linearcde(X)

    # Set model back to evaluation
    a5linearcde.eval()
    output_no_dropout_again = a5linearcde(X)

    # Set model back to training
    a5linearcde.train()
    output_with_dropout_again = a5linearcde(X)

    # Check that including dropout impacts the output
    assert not torch.equal(output_no_dropout, output_with_dropout)
    # Check that the model is deterministic with dropout turned off
    assert torch.equal(output_no_dropout, output_no_dropout_again)
    # Check that the model is stochastic with dropout turned on
    assert not torch.equal(output_with_dropout, output_with_dropout_again)
