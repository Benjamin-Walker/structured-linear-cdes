import datetime
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
from data_dir.dataloaders import (
    create_a5_dataloaders,
    create_fl_dataloaders,
    create_snake_dataloaders,
)
from models.lr_scheduler import LinearWarmupCosineAnnealing


def train_model(
    config,
    data_dim,
    label_dim,
    task,
    model_name,
    model,
    dataloader,
    num_steps,
    print_steps,
    learning_rate,
    device,
    early_stop_threshold=1.0,
    weight_decay_embedding=0.0,
    weight_decay_others=1e-2,
    warmup_fraction=0.1,
    final_lr=1e-5,
):
    """
    Trains the model for a specified number of steps, logging progress, and optionally
    performing early stopping based on validation accuracy.

    Args:
        config (dict): Configuration parameters for training.
        data_dim (int): Dimensionality of the input data.
        label_dim (int): Dimensionality of the labels.
        task (str): Task name (e.g. A5).
        model_name (str): Name of the model.
        model (nn.Module): The model to be trained.
        dataloader (dict[str, DataLoader or Iterable]): Must contain "train" and "val" keys.
        num_steps (int): Total steps to train.
        print_steps (int): Frequency of logging and validation evaluation.
        learning_rate (float): Base learning rate for training.
        device (torch.device): Device for computation (CPU or GPU).
        early_stop_threshold (float, optional): Validation accuracy threshold for early stopping.
        weight_decay_embedding (float, optional): Weight decay for embedding parameters.
        weight_decay_others (float, optional): Weight decay for all other parameters.
        warmup_fraction (float, optional): Fraction of total steps for linear warmup.
        final_lr (float, optional): Final learning rate for cosine annealing.

    Returns:
        model (nn.Module): The trained model.
        steps (list[int]): Steps at which logs were printed.
        val_accs (list[float]): Validation accuracies per print step.
        early_stop (bool): Whether early stopping was triggered.
    """
    model.to(device)
    embedding_params = [p for n, p in model.named_parameters() if "embedding" in n]
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]

    optimizer = optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": weight_decay_embedding},
            {"params": other_params, "weight_decay": weight_decay_others},
        ],
        lr=learning_rate,
    )

    warmup_steps = int(warmup_fraction * num_steps)
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer, warmup_steps, num_steps, learning_rate, final_lr
    )

    criterion = nn.CrossEntropyLoss()
    step = 0
    total_loss = 0
    steps = []
    val_accs = []
    start_time = time.time()

    model.train()
    while step < num_steps:
        for X, y, mask in dataloader["train"]:
            optimizer.zero_grad()

            # Handle the case where we have multi-length data in a single batch
            if isinstance(X, tuple):
                X, X_2 = X
                y, y_2 = y
                mask, mask_2 = mask

                X_2, y_2 = X_2.to(device), y_2.to(device)
                outputs_2 = model(X_2)
                loss = criterion(outputs_2[mask_2], y_2[mask_2].flatten())
                loss.backward()

            X, y, mask = X.to(device), y.to(device), mask.to(device)
            outputs = model(X)
            loss = criterion(outputs[mask], y[mask].flatten())
            loss.backward()

            # Zero out any gradients if your model requires it
            model.mask_grads()

            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

            # Logging
            if step % print_steps == 0:
                model.eval()
                end_time = time.time()
                val_acc = 0
                val_num = 0

                with torch.no_grad():
                    for X_val, y_val, mask_val in dataloader["val"]:
                        X_val, y_val, mask_val = (
                            X_val.to(device),
                            y_val.to(device),
                            mask_val.to(device),
                        )
                        outputs_val = model(X_val)
                        val_acc += (
                            (
                                outputs_val[mask_val].argmax(dim=-1)
                                == y_val[mask_val].flatten()
                            )
                            .sum()
                            .item()
                        )
                        val_num += y_val[mask_val].flatten().size(0)

                if step == 0 and print_steps > 1:
                    total_loss *= print_steps

                avg_loss = total_loss / print_steps
                accuracy = val_acc / val_num if val_num > 0 else 0
                print(
                    f"Step: {step}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Validation Acc: {accuracy:.4f}, "
                    f"Time: {end_time - start_time:.2f} sec"
                )

                steps.append(step)
                val_accs.append(accuracy)

                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Save model checkpoint
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "data_dim": data_dim,
                    "label_dim": label_dim,
                    "model_name": model_name,
                    "final_val_acc": val_accs[-1] if val_accs else None,
                    "timestamp": timestamp_str,
                }

                checkpoint_filename = f"checkpoint_{task}_{model_name}.pt"
                checkpoint_path = os.path.join("checkpoints", checkpoint_filename)
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved model checkpoint to: {checkpoint_path}")

                early_stop = accuracy > early_stop_threshold

                out_filename = f"results_{task}_{model_name}.json"
                out_path = os.path.join("results", out_filename)

                # Gather all relevant info to save:
                results_dict = {
                    "config": config,
                    "steps": steps,
                    "val_accs": val_accs,
                    "early_stop": early_stop,
                }

                with open(out_path, "w") as f:
                    json.dump(results_dict, f, indent=2)

                print(f"Saved results to: {out_path}")

                start_time = time.time()
                total_loss = 0
                model.train()

                if early_stop:
                    return model, steps, val_accs, True

            if step >= num_steps:
                break

            step += 1

    return model, steps, val_accs, False


def run_experiment(config):
    """
    Main driver for the experiment. Creates dataloaders (either for A5 or formal language),
    creates the model, calls train_model(), and saves results to the results/ folder.
    """
    # Read config fields
    task = config["task"]  # e.g., "A5" or "majority"
    model_name = config["model_name"]  # e.g., "lcde" or "mamba"
    num_blocks = config["num_blocks"]
    model_dim = config["model_dim"]
    batch_size = config["batch_size"]
    num_steps = config["num_steps"]
    print_steps = config["print_steps"]
    learning_rate = config["learning_rate"]

    # Optional fields
    diagonal = config.get("diagonal", False)
    fwht = config.get("fwht", False)
    use_glu = config.get("use_glu", False)
    second_embedding = config.get("second_embedding", False)
    early_stop_threshold = config.get("early_stop_threshold", 1.0)
    init_std = config.get("init_std", 1.0)
    sparsity = config.get("sparsity", 1.0)
    dropout_rate = config.get("dropout_rate", 0.01)
    length = config.get("length")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoader(s)
    if task == "A5":
        train_dataloader, val_dataloader, data_dim, label_dim = create_a5_dataloaders(
            length=length, train_split=0.8, batch_size=batch_size
        )

        # Optional: combining with length=2 if needed
        dataloader_length2, _, _, _ = create_a5_dataloaders(
            2, train_split=1.0, batch_size=batch_size // 10
        )

        def train_dataloader_multilength():
            while True:
                for (X, y, mask), (X_2, y_2, mask_2) in zip(
                    train_dataloader, dataloader_length2
                ):
                    yield (X, X_2), (y, y_2), (mask, mask_2)

        dataloader = {"train": train_dataloader_multilength(), "val": val_dataloader}
    elif task == "snake":
        train_dataloader, val_dataloader, data_dim, label_dim = (
            create_snake_dataloaders(
                pt_file="data_dir/snake_games/big_snake.pt",
                batch_size=batch_size,
                train_split=0.999,
            )
        )

        dataloader = {"train": train_dataloader, "val": val_dataloader}
    else:
        # Formal language tasks, e.g. "majority"
        train_dataloader, _, data_dim, label_dim = create_fl_dataloaders(
            task,
            num_samples=25600000,
            batch_size=batch_size,
            min_length=3,
            max_length=40,
            padding_length=256,
            train_split=1.0,
            seed=1234,
        )
        val_dataloader, _, _, _ = create_fl_dataloaders(
            task,
            num_samples=8192,
            batch_size=batch_size,
            min_length=40,
            max_length=256,
            padding_length=256,
            train_split=1.0,
            seed=2345,
        )
        dataloader = {"train": train_dataloader, "val": val_dataloader}

    # Create the model
    if model_name == "mamba":
        from models.mamba import StackedMamba

        model = StackedMamba(
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            second_embedding=second_embedding,
        )
    elif model_name == "lcde":
        from models.slcde import StackedLCDE

        model = StackedLCDE(
            num_blocks=num_blocks,
            model_dim=model_dim,
            data_dim=data_dim,
            label_dim=label_dim,
            init_std=init_std,
            sparsity=sparsity,
            dropout_rate=dropout_rate,
            use_glu=use_glu,
            diagonal=diagonal,
            fwht=fwht,
            second_embedding=second_embedding,
        )
    elif model_name == "lstm":
        from models.lstm import LSTM

        model = LSTM(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
        )
    elif model_name == "xlstm":
        from models.xlstm import xLSTM

        model = xLSTM(
            num_blocks=num_blocks,
            data_dim=data_dim,
            model_dim=model_dim,
            label_dim=label_dim,
            dropout_rate=dropout_rate,
            second_embedding=second_embedding,
            context_length=260,
        )
    else:
        raise ValueError(
            "Model not recognized. Must be 'mamba', 'lcde', 'lstm', or 'xlstm'."
        )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {pytorch_total_params}")

    # Create directories for results and checkpoints
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Train
    model, steps, val_accs, early_stop = train_model(
        config=config,
        data_dim=data_dim,
        label_dim=label_dim,
        task=task,
        model_name=model_name,
        model=model,
        dataloader=dataloader,
        num_steps=num_steps,
        print_steps=print_steps,
        learning_rate=learning_rate,
        early_stop_threshold=early_stop_threshold,
        device=device,
    )

    if early_stop:
        print("Early stop triggered! Finished before reaching num_steps.")
    else:
        print("Training complete. Did not trigger early stopping.")
