import torch

from models.mamba import MambaBlock, StackedMamba
from models.slcde import LinearCDE, StackedLCDE

torch.manual_seed(1234)


def test_mamba_block_forward():
    """
    Basic test for the MambaBlock forward pass.
    Assumes MambaBlock(d_model=input_dim), returns the same shape as input.
    """
    batch_size = 2
    seq_len = 3
    input_dim = 256
    dropout_rate = 0.1

    # Ensure test runs on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block = MambaBlock(input_dim, dropout_rate=dropout_rate).to(device)
    x = torch.randn(batch_size, seq_len, input_dim).to(device)

    # Perform forward pass
    out = block(x)

    # Assert output shape matches input shape
    assert out.shape == (batch_size, seq_len, input_dim)


def test_stacked_mamba_forward():
    """
    Basic test for StackedMamba forward pass.
    Assumes StackedMamba applies multiple MambaBlocks in sequence,
    plus a final linear layer mapping from model_dim -> label_dim.
    """
    batch_size = 2
    seq_len = 3
    num_blocks = 2
    model_dim = 256
    data_dim = 10  # Used for nn.Embedding in StackedMamba
    label_dim = 5
    dropout_rate = 0.1

    # Ensure test runs on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StackedMamba(
        num_blocks=num_blocks,
        model_dim=model_dim,
        data_dim=data_dim,
        label_dim=label_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    # Input of token IDs (batch_size, seq_len)
    X = torch.randint(0, data_dim, (batch_size, seq_len)).to(device)

    # Perform forward pass
    out = model(X)

    # Assert output shape matches expected shape
    assert out.shape == (batch_size, seq_len, label_dim)


def test_linearcde_forward():
    batch_size = 2
    seq_len = 3
    input_dim = 4
    hidden_dim = 5

    linear_cde = LinearCDE(input_dim, hidden_dim)

    # Input tensor (batch_size, seq_len, input_dim)
    X = torch.randn(batch_size, seq_len, input_dim)

    # Perform forward pass
    output = linear_cde(X)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, hidden_dim)


def test_stackedlcde_forward():
    batch_size = 2
    seq_len = 3
    num_blocks = 2
    model_dim = 10
    data_dim = 4
    label_dim = 6

    # The stacked LCDE expects an input of shape (batch_size, seq_len, data_dim)
    model = StackedLCDE(
        num_blocks=num_blocks,
        model_dim=model_dim,
        data_dim=data_dim,
        label_dim=label_dim,
    )

    # Input of token IDs (batch_size, seq_len)
    X = torch.randint(0, data_dim, (batch_size, seq_len))

    # Perform forward pass
    output = model(X)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, label_dim)


def test_increment_linearcde():
    input_dim = 1
    hidden_dim = 3

    linear_cde = LinearCDE(input_dim, hidden_dim)
    with torch.no_grad():
        linear_cde.init_layer.weight.data.fill_(0)
        linear_cde.vf_B.weight.fill_(0)
        bias_init = torch.zeros(hidden_dim)
        bias_init[0] = 1
        linear_cde.init_layer.bias = torch.nn.Parameter(bias_init)

        vf_A = torch.zeros(hidden_dim * hidden_dim, input_dim + 1)
        vf_A[3, 0] = 1
        vf_A[6, 1] = 1
        linear_cde.vf_A.weight = torch.nn.Parameter(vf_A)

    X = torch.Tensor([[[0], [1.5], [-1.5]]])

    output = linear_cde(X)

    # Linear CDE with the given initialisation should give the increment of the
    # two dimensional path with the first dimension as time from 0 to 1 and
    # the second dimension as cumulative sum of the input
    expected = torch.Tensor([[[1, 0, 0], [1, 0.5, 1.5], [1, 1.0, 0.0]]])
    assert torch.all(output.eq(expected))


def test_stackedlcde_dropout():
    batch_size = 2
    seq_len = 3
    num_blocks = 2
    model_dim = 8
    data_dim = 4
    label_dim = 6

    model = StackedLCDE(
        num_blocks=num_blocks,
        model_dim=model_dim,
        data_dim=data_dim,
        label_dim=label_dim,
        dropout_rate=0.1,
    )

    # Input tensor (batch_size, seq_len, data_dim)
    X = torch.randint(0, data_dim, (batch_size, seq_len))

    # Set model to evaluation mode
    model.eval()
    output_no_dropout = model(X)

    # Set model to training mode
    model.train()
    output_with_dropout = model(X)

    # Set model back to evaluation
    model.eval()
    output_no_dropout_again = model(X)

    # Set model back to training
    model.train()
    output_with_dropout_again = model(X)

    # Check that including dropout impacts the output
    assert not torch.equal(output_no_dropout, output_with_dropout)
    # Check that the model is deterministic with dropout turned off
    assert torch.equal(output_no_dropout, output_no_dropout_again)
    # Check that the model is stochastic with dropout turned on
    assert not torch.equal(output_with_dropout, output_with_dropout_again)


def test_linearcde_init_std():
    input_dim = 20
    hidden_dim = 40
    init_std = 0.5

    init_layer_weight_stds = []
    init_layer_bias_stds = []
    vf_B_weight_stds = []
    vf_A_stds = []

    # We'll create multiple LinearCDE instances to gather stats
    for _ in range(100):
        linear_cde = LinearCDE(input_dim, hidden_dim, init_std=init_std)
        init_layer_weight_stds.append(linear_cde.init_layer.weight.std().item())
        init_layer_bias_stds.append(linear_cde.init_layer.bias.std().item())
        vf_B_weight_stds.append(linear_cde.vf_B.weight.std().item())
        vf_A_stds.append(linear_cde.vf_A.weight.std().item())

    # Check the approximate means of the distributions
    assert torch.isclose(
        torch.tensor(init_layer_weight_stds).mean(), torch.tensor(init_std), atol=0.01
    )
    assert torch.isclose(
        torch.tensor(init_layer_bias_stds).mean(), torch.tensor(init_std), atol=0.01
    )
    assert torch.isclose(
        torch.tensor(vf_B_weight_stds).mean(), torch.tensor(init_std), atol=0.01
    )
    assert torch.isclose(
        torch.tensor(vf_A_stds).mean(),
        torch.tensor(init_std / (hidden_dim**0.5)),
        atol=0.01,
    )


def test_linearcde_sparsity():
    input_dim = 20
    hidden_dim = 40

    # Test with different levels of sparsity
    for sparsity in [0.0, 0.5, 1.0]:
        if sparsity == 0.0:
            linear_cde = LinearCDE(input_dim, hidden_dim, sparsity=sparsity)
            mask = linear_cde.mask
            # Expecting all zeros
            assert torch.equal(mask, torch.zeros_like(mask, dtype=torch.bool))
        elif sparsity == 1.0:
            linear_cde = LinearCDE(input_dim, hidden_dim, sparsity=sparsity)
            mask = linear_cde.mask
            # Expecting all ones
            assert torch.equal(mask, torch.ones_like(mask, dtype=torch.bool))
        else:
            linear_cde = LinearCDE(input_dim, hidden_dim, sparsity=sparsity)
            mask = linear_cde.mask
            # Expecting the average sparsity level to be close to the target
            # (1 - mean(mask)) ≈ sparsity
            actual_sparsity = 1.0 - mask.float().mean().item()
            assert torch.isclose(
                torch.tensor(actual_sparsity), torch.tensor(sparsity), atol=0.01
            )
