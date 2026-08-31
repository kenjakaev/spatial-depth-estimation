from torch.utils.data import DataLoader
from src.data import NYUDepthDataset


def get_dataloaders(
    base_dir, train_csv, test_csv, batch_size, num_workers=4, img_size=(384, 384)
):
    train_dataset = NYUDepthDataset(
        base_dir, train_csv, img_size=img_size, is_train=True
    )
    test_dataset = NYUDepthDataset(
        base_dir, test_csv, img_size=img_size, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader
