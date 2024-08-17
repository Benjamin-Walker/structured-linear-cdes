import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data_dir.dataloaders import create_a5_dataloaders
from models.slcde import A5LinearCDE


def train_model(
    model, dataloader_length2, dataloader, num_steps, print_steps, learning_rate
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    step = 0
    total_loss = 0
    steps = []
    test_accs = []
    start_time = time.time()

    while step < num_steps:
        for X, y in dataloader_length2:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.permute(0, 2, 1), y)
            loss.backward()
            optimizer.step()
        for X, y in dataloader["train"]:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.permute(0, 2, 1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % print_steps == 0:
                model.eval()
                end_time = time.time()
                with torch.no_grad():
                    train_acc = 0
                    train_num = 0
                    for _, (X, y) in zip(range(5), dataloader["train"]):
                        outputs = model(X)
                        train_acc += (outputs.argmax(dim=-1) == y).sum().item()
                        train_num += y.size(0) * y.size(1)
                    test_acc = 0
                    test_num = 0
                    for _, (X, y) in zip(range(5), dataloader["test"]):
                        outputs = model(X)
                        test_acc += (outputs.argmax(dim=-1) == y).sum().item()
                        test_num += y.size(0) * y.size(1)
                if step == 0:
                    total_loss *= print_steps
                print(
                    f"Step {step}, Loss: {total_loss / print_steps:.4f}, Train Acc: {train_acc / train_num:.4f}, Test Acc: {test_acc / test_num:.4f}, Time: {end_time - start_time:.2f}s"
                )
                steps.append(step)
                test_accs.append(test_acc / test_num)
                start_time = time.time()
                total_loss = 0
                model.train()

            step += 1
            if step >= num_steps:
                break

    return model, steps, test_acc


if __name__ == "__main__":
    torch.manual_seed(5678)

    if not os.path.isdir("outputs"):
        os.mkdir("outputs")

    hidden_dim = 110
    label_dim = 60
    batch_size = 32
    data_dim = 255
    omega_dim = data_dim + 1
    xi_dim = data_dim + 1
    length = 20

    model = A5LinearCDE(hidden_dim, data_dim, omega_dim, xi_dim, label_dim)

    train_dataloader, test_dataloader = create_a5_dataloaders(
        length, train_split=0.8, batch_size=batch_size
    )
    dataloader_length2, _ = create_a5_dataloaders(
        2, train_split=1.0, batch_size=batch_size
    )

    dataloader = {"train": train_dataloader, "test": test_dataloader}

    model, steps, test_accs = train_model(
        model,
        dataloader_length2,
        dataloader,
        num_steps=1000000,
        print_steps=10000,
        learning_rate=3e-4,
    )
