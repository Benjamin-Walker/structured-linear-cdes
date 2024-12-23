import time

import torch
import torch.nn as nn
import torch.optim as optim

from data_dir.dataloaders import create_a5_dataloaders, create_fl_dataloaders
from models.lr_scheduler import LinearWarmupCosineAnnealing
from models.mamba import StackedMamba
from models.slcde import StackedLCDE


def train_model(model, dataloader, num_steps, print_steps, learning_rate, device):
    model.to(device)
    embedding_params = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]
    optimizer = optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": 1e-2},  # or your chosen value
        ],
        lr=learning_rate,
    )
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        0.1 * num_steps,
        num_steps,
        learning_rate,
        1e-5,
    )
    criterion = nn.CrossEntropyLoss()

    step = 0
    total_loss = 0
    steps = []
    val_accs = []
    start_time = time.time()

    while step < num_steps:
        for X, y, mask in dataloader["train"]:
            optimizer.zero_grad()

            # Handle the case where we have two lengths in the dataloader
            if isinstance(X, tuple):
                X, X_2 = X
                y, y_2 = y
                mask, mask_2 = mask
                X_2, y_2 = X_2.to(device), y_2.to(device)
                outputs_2 = model(X_2)
                loss = criterion(outputs_2[mask_2], y_2[mask_2].flatten())
                loss.backward()

            X, y, mask = X.to(device), y.to(device), mask.to(device)
            # X[:, 1:][mask[:, :-1]] = y[mask]
            outputs = model(X)
            loss = criterion(outputs[mask], y[mask].flatten())
            loss.backward()
            model.mask_grads()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

            if step % print_steps == 0:
                model.eval()
                end_time = time.time()
                val_acc = 0
                val_num = 0
                with torch.no_grad():
                    for X, y, mask in dataloader["val"]:
                        X, y, mask = X.to(device), y.to(device), mask.to(device)
                        # X[:, 1:][mask[:, :-1]] = y[mask]
                        outputs = model(X)
                        val_acc += (
                            (outputs[mask].argmax(dim=-1) == y[mask].flatten())
                            .sum()
                            .item()
                        )
                        val_num += y[mask].flatten().size(0)
                if step == 0:
                    # If step == 0, we've effectively only done 1 iteration so far
                    # but we scale total_loss as if we had print_steps iters
                    total_loss *= print_steps

                print(
                    f"Step: {step}, "
                    f"Loss: {total_loss / print_steps:.4f}, "
                    f"Validation Acc: {val_acc / val_num:.4f}, "
                    f"Time: {end_time - start_time:.2f} sec"
                )
                steps.append(step)
                val_accs.append(val_acc / val_num)
                start_time = time.time()
                total_loss = 0
                model.train()

                if val_acc / val_num > 0.9:
                    return model, steps, val_accs, True

            if step >= num_steps:
                break

            step += 1

    return model, steps, val_accs, False


if __name__ == "__main__":
    # task_list = [
    #     "bucket_sort",
    #     "cycle_nav",
    #     "even_pairs",
    #     "majority",
    #     "majority_count",
    #     "mod_arith_no_brack",
    #     "mod_arith_w_brack",
    #     "missing_duplicate",
    #     "odds_first",
    #     "parity",
    #     "repetition",
    #     "reverse_string",
    #     "set",
    #     "solve_equation",
    #     "stack_manipulation",
    # ]

    # random_guess = {
    #     "bucket_sort": 1 / 10,
    #     "cycle_nav": 1 / 5,
    #     "even_pairs": 1 / 2,
    #     "majority": 1 / 63,
    #     "majority_count": 1 / 63,
    #     "missing_duplicate": 1 / 9,
    #     "mod_arith_no_brack": 1 / 5,
    #     "mod_arith_w_brack": 1 / 5,
    #     "odds_first": 1 / 10,
    #     "parity": 1 / 2,
    #     "repetition": 1 / 10,
    #     "reverse_string": 1 / 10,
    #     "set": 1 / 10,
    #     "solve_equation": 1 / 5,
    #     "stack_manipulation": 1 / 4,
    # }

    task = "A5"
    lengths = list(range(3, 21))
    depths = list(range(1, 5))

    model_name = "lcde"
    num_blocks = 1
    batch_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_val_accs = {}
    best_scaled_val_accs = {}

    depth_recurrence = depths.copy()

    for length in lengths:
        depths_to_remove = []
        for depth in depth_recurrence:
            print(f"Length: {length}, Depth: {depth}")
            if task == "A5":
                train_dataloader, val_dataloader, data_dim, label_dim = (
                    create_a5_dataloaders(
                        length, train_split=0.8, batch_size=batch_size
                    )
                )
                dataloader_length2, _, _, _ = create_a5_dataloaders(
                    2, train_split=1.0, batch_size=batch_size // 10
                )

                def train_dataloader_multilength():
                    while True:
                        for (X, y), (X_2, y_2) in zip(
                            train_dataloader, dataloader_length2
                        ):
                            yield (X, X_2), (y, y_2)
            else:
                train_dataloader, _, data_dim, label_dim = create_fl_dataloaders(
                    task,
                    num_samples=25600000,
                    batch_size=batch_size,
                    min_length=3,
                    max_length=40,
                    padding_length=260,
                    train_split=1.0,
                    seed=1234,
                )

                val_dataloader, _, _, _ = create_fl_dataloaders(
                    task,
                    num_samples=8192,
                    batch_size=batch_size,
                    min_length=40,
                    max_length=256,
                    padding_length=260,
                    train_split=1.0,
                    seed=2345,
                )

            if model_name == "mamba":
                model_dim = 512
                model = StackedMamba(
                    num_blocks, model_dim, data_dim, label_dim, use_glu=True
                )
            elif model_name == "lcde":
                model_dim = 128
                model = StackedLCDE(
                    depth,
                    model_dim,
                    data_dim,
                    model_dim,
                    label_dim,
                    init_std=1 / 20,
                    use_glu=True,
                    diagonal=True,
                    fwht=True,
                )
            else:
                raise ValueError("Model not recognized")

            dataloader = {"train": train_dataloader, "val": val_dataloader}

            model, steps, val_accs, early_stop = train_model(
                model,
                dataloader,
                num_steps=1000000,
                print_steps=10000,
                learning_rate=2e-4,
                device=device,
            )

            if early_stop:
                break
            else:
                depths_to_remove.append(depth)

        for depth in depths_to_remove:
            depth_recurrence.remove(depth)
