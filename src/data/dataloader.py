from torch.utils.data import DataLoader
from src.data.dataset import NYUDepthDataset


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


if __name__ == "__main__":
    from src.config import DATA_DIR, BASE_DIR

    csv_train = DATA_DIR / "nyu2_train.csv"
    csv_test = DATA_DIR / "nyu2_test.csv"

    train, test = get_dataloaders(BASE_DIR, csv_train, csv_test, 32)

    train_image, train_depth = next(iter(train))
    test_image, test_depth = next(iter(test))

    print(
        f"Train image shape: {train_image.shape} | Train depth shape: {train_depth.shape}"
    )
    print(
        f"Test image shape: {test_image.shape} | Test depth shape: {test_depth.shape}"
    )
