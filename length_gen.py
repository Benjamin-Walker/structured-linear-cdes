import argparse
import os

import torch

from data_dir.dataloaders import create_group_dataloaders


def build_model(model_name, config, data_dim, label_dim, device):
    """
    Build and return a model instance based on 'model_name' & 'config'.
    """

    if model_name == "mamba":
        from models.mamba import StackedMamba

        model = StackedMamba(
            num_blocks=config["num_blocks"],
            model_dim=config["model_dim"],
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=config.get("dropout_rate", 0.01),
            use_glu=config.get("use_glu", False),
            second_embedding=config.get("second_embedding", False),
        )
    elif model_name == "s6":
        from models.s6 import S6

        model = S6(
            num_blocks=config["num_blocks"],
            model_dim=config["model_dim"],
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=config.get("dropout_rate", 0.01),
            use_glu=config.get("use_glu", False),
            second_embedding=config.get("second_embedding", False),
            d_state=config.get("d_state", 16),
            dt_rank=config.get("dt_rank", "auto"),
        )

    elif model_name == "transformer":
        from models.transformer import CausalTransformer

        model = CausalTransformer(
            num_blocks=config["num_blocks"],
            model_dim=config["model_dim"],
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=config.get("dropout_rate", 0.01),
            second_embedding=config.get("second_embedding", False),
        )

    elif model_name in ["deltanet", "gateddeltanet", "deltaproduct", "rwkv6", "rwkv7"]:
        from models.fla import StackedBlock

        model = StackedBlock(
            layer_type=model_name,  # "deltanet", "gateddeltanet", "rwkv6", or "rwkv7"
            num_blocks=config["num_blocks"],
            model_dim=config["model_dim"],
            data_dim=data_dim,
            label_dim=label_dim,
            rank=config.get("rank", 0),
            gated=config.get("gated", True),
            dropout_rate=config.get("dropout_rate", 0.01),
            use_glu=config.get("use_glu", False),
            second_embedding=config.get("second_embedding", False),
        )

    elif model_name == "deltanet2":
        from models.deltanet2 import StackedBlock as StackedDeltaNet2

        model = StackedDeltaNet2(
            num_blocks=config["num_blocks"],
            model_dim=config["model_dim"],
            data_dim=data_dim,
            label_dim=label_dim,
            sigmoid_scale=2,
            dropout_rate=config.get("dropout_rate", 0.01),
            use_glu=config.get("use_glu", False),
            second_embedding=config.get("second_embedding", False),
        )

    elif model_name == "lcde":
        from models.slcde import StackedLCDE

        model = StackedLCDE(
            num_blocks=config["num_blocks"],
            model_dim=config["model_dim"],
            data_dim=data_dim,
            label_dim=label_dim,
            init_std=config.get("init_std", 1.0),
            block_size=config.get("block_size", 1),
            sparsity=config.get("sparsity", 1.0),
            dropout_rate=config.get("dropout_rate", 0.01),
            use_glu=config.get("use_glu", False),
            diagonal=config.get("diagonal", False),
            diagonal_dense=config.get("diagonal_dense", False),
            fwht=config.get("fwht", False),
            second_embedding=config.get("second_embedding", False),
            rank=config.get("rank", 0),
        )

    elif model_name == "lstm":
        from models.lstm import LSTM

        model = LSTM(
            num_blocks=config["num_blocks"],
            data_dim=data_dim,
            model_dim=config["model_dim"],
            label_dim=label_dim,
            dropout_rate=config.get("dropout_rate", 0.01),
            second_embedding=config.get("second_embedding", False),
        )

    elif model_name == "xlstm":
        from models.xlstm import xLSTM

        model = xLSTM(
            num_blocks=config["num_blocks"],
            data_dim=data_dim,
            model_dim=config["model_dim"],
            label_dim=label_dim,
            slstm_at=config.get("slstm_at", [1]),
            dropout_rate=config.get("dropout_rate", 0.01),
            second_embedding=config.get("second_embedding", False),
            context_length=config.get("padding_length", 256),
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.to(device)
    return model


def evaluate_model_on_length(
    model, seq_length, batch_size, num_samples, token_accuracy=False, device="cpu"
):
    """
    Creates a dataloader for 'task' sequences of exactly seq_length,
    evaluates model on it, and returns the accuracy.
    """
    val_dataloader, _, data_dim, label_dim = create_group_dataloaders(
        group="A5",
        num_samples=num_samples,
        batch_size=batch_size,
        min_length=seq_length,
        max_length=seq_length,
        padding_length=seq_length,
        train_split=1.0,
        seed=123465,
    )

    model.eval()
    correct = 0
    total = 0
    if token_accuracy:
        with torch.no_grad():
            for X_val, y_val, mask_val in val_dataloader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                mask_val = mask_val.to(device)

                outputs_val = model(X_val)  # (batch, seq, label_dim)
                pred = outputs_val[mask_val].argmax(dim=-1)
                labels = y_val[mask_val].flatten()

                correct += (pred == labels).sum().item()
                total += labels.size(0)
    else:
        with torch.no_grad():
            for X_val, y_val, mask_val in val_dataloader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                mask_val = mask_val.to(device)

                logits = model(X_val)
                token_correct = (logits.argmax(-1) == y_val) & mask_val  # (batch, seq)
                seq_correct = token_correct.all(dim=1)  #    (batch,)
                correct += seq_correct.sum().item()
                total += seq_correct.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint .pt file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "xlstm",
            "s6",
            "mamba",
            "lstm",
            "deltanet",
            "deltanet2",
            "gateddeltanet",
            "deltaproduct",
            "rwkv6",
            "rwkv7",
            "lcde",
            "transformer",
        ],
        help="Which model to load (must match how it was originally trained).",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="40,56,72,88,104,120,136,152,168,184,200,216,232,248",
        help="Comma-separated list of sequence lengths to test.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2048,
        help="Number of sequences to sample for each length.",
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="If set, uses strict=True when loading state_dict.",
    )
    parser.add_argument(
        "--token_accuracy",
        action="store_true",
        help="If set, evaluates token accuracy instead of sequence accuracy.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_sd = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    data_dim = checkpoint["data_dim"]
    label_dim = checkpoint["label_dim"]

    model = build_model(args.model_name, config, data_dim, label_dim, device)

    if args.strict_load:
        model.load_state_dict(model_sd, strict=True)
    else:
        model.load_state_dict(model_sd, strict=False)

    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"Evaluating '{args.model_name}' on group='A5' for lengths={lengths}...")

    for L in lengths:
        acc = evaluate_model_on_length(
            model,
            seq_length=L,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            token_accuracy=args.token_accuracy,
            device=device,
        )
        print(f"Length={L}: Accuracy={acc:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
