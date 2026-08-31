import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_img(
    img_tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C
    mean = np.array(mean)
    std = np.array(std)
    img = std * img + mean
    return np.clip(img, 0, 1)


def draw_sample(model, test_loader, device="cpu"):
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            preds = model(X_test)
            preds = torch.clamp(preds, min=0.0)
            break

    img = denormalize_img(X_test[0])
    gt_depth = y_test[0].cpu().squeeze().numpy() * 10.0
    pred_depth = preds[0].cpu().squeeze().numpy() * 10.0

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title("Input RGB")
    ax[1].imshow(gt_depth, cmap="magma")
    ax[1].set_title("Ground Truth Depth")
    ax[2].imshow(pred_depth, cmap="magma")
    ax[2].set_title("Predicted Depth")
    plt.show()
