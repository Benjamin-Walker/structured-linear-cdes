import time

import torch
import torch.nn as nn
import torch.optim as optim

from data_dir.dataloaders import create_a5_dataloaders
from models.mamba import StackedMamba
from models.slcde import StackedLCDE


def train_model(model, dataloader, num_steps, print_steps, learning_rate, device):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    step = 0
    total_loss = 0
    steps = []
    test_accs = []
    start_time = time.time()

    while step < num_steps:
        for X, y in dataloader["train"]:
            optimizer.zero_grad()

            # Handle the case where we have two lengths in the dataloader
            if isinstance(X, tuple):
                X, X_2 = X
                y, y_2 = y
                X_2, y_2 = X_2.to(device), y_2.to(device)
                outputs_2 = model(X_2)
                loss = criterion(outputs_2.permute(0, 2, 1), y_2)
                loss.backward()

            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs.permute(0, 2, 1), y)
            loss.backward()
            model.mask_grads()
            optimizer.step()
            total_loss += loss.item()

            if step % print_steps == 0:
                model.eval()
                end_time = time.time()
                test_acc = 0
                test_num = 0
                with torch.no_grad():
                    for X, y in dataloader["test"]:
                        X, y = X.to(device), y.to(device)
                        outputs = model(X)
                        test_acc += (outputs.argmax(dim=-1) == y).sum().item()
                        test_num += y.size(0) * y.size(1)

                if step == 0:
                    # If step == 0, we've effectively only done 1 iteration so far
                    # but we scale total_loss as if we had print_steps iters
                    total_loss *= print_steps

                print(
                    f"Step: {step}, "
                    f"Loss: {total_loss / print_steps:.4f}, "
                    f"Test Acc: {test_acc / test_num:.4f}, "
                    f"Time: {end_time - start_time:.2f} sec"
                )
                steps.append(step)
                test_accs.append(test_acc / test_num)
                start_time = time.time()
                total_loss = 0
                model.train()

            if step >= num_steps:
                break

            step += 1

    return model, steps, test_accs


if __name__ == "__main__":
    model_name = "mamba"
    length = 20
    data_dim = 60
    label_dim = 60
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "mamba":
        model_dim = 512
        model = StackedMamba(4, model_dim, data_dim, label_dim)
    elif model_name == "lcde":
        model_dim = 128
        model = StackedLCDE(
            1, model_dim, data_dim, model_dim, label_dim, init_std=1 / 20
        )
    else:
        raise ValueError("Model not recognized")

    train_dataloader, test_dataloader = create_a5_dataloaders(
        length, train_split=0.8, batch_size=batch_size
    )
    dataloader_length2, _ = create_a5_dataloaders(
        2, train_split=1.0, batch_size=batch_size // 10
    )

    def train_dataloader_multilength():
        while True:
            for (X, y), (X_2, y_2) in zip(train_dataloader, dataloader_length2):
                yield (X, X_2), (y, y_2)

    dataloader = {"train": train_dataloader_multilength(), "test": test_dataloader}

    model, steps, test_accs = train_model(
        model,
        dataloader,
        num_steps=1000000,
        print_steps=1000,
        learning_rate=1e-4,
        device=device,
    )
