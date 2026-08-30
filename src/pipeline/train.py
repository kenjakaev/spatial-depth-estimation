import torch


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = "cuda",
):
    model.to(device)
    model.train()

    train_loss = 0.0

    scaler = torch.amp.GradScaler(device=device)

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        train_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_loss /= len(data_loader)
    print(f"Train Loss: {train_loss:.5f}")
    return train_loss


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device = "cuda",
):
    model.to(device)
    model.eval()

    test_loss = 0.0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

    test_loss /= len(data_loader)
    print(f"Test Loss: {test_loss:.5f}\n")
    return test_loss


def predict_depth(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = "cuda",
):
    model.to(device)
    model.eval()

    predictions = []

    with torch.inference_mode():
        for X, _ in data_loader:
            X = X.to(device)
            y_pred = model(X)

            y_pred = torch.clamp(y_pred, min=0.0)

            predictions.append(y_pred.cpu())

    return torch.cat(predictions, dim=0)
