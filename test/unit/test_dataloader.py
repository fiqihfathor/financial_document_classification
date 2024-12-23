import pytest
from torch.utils.data import DataLoader
from src.ml.preprocessing.dataloader import get_image_dataloader
from src.ml.preprocessing.dataset import DocumentDataset
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_dataset():
    class MockDataset:
        def __init__(self):
            self.data = [f"image_{i}" for i in range(100)]
            self.labels = [i % 2 for i in range(100)]

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def __len__(self):
            return len(self.data)

    return MockDataset()

@patch("src.ml.preprocessing.dataloader.DocumentDataset")
def test_dataloader_ratios(mock_image_dataset, mock_dataset):
    mock_image_dataset.return_value = mock_dataset

    train_loader, val_loader, test_loader = get_image_dataloader(
        root_dir="dummy_path",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=10,
        shuffle=True
    )

    total_len = len(mock_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    assert len(train_loader.dataset) == pytest.approx(train_len, abs=1)
    assert len(val_loader.dataset) == pytest.approx(val_len, abs=1)
    assert len(test_loader.dataset) == pytest.approx(test_len, abs=1)
@patch("src.ml.preprocessing.dataloader.DocumentDataset")
def test_dataloader_stratification(mock_image_dataset,mock_dataset):
    mock_image_dataset.return_value = mock_dataset

    train_loader, val_loader, test_loader = get_image_dataloader(
        mock_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=10,
        shuffle=True
    )

    train_labels = [mock_dataset.labels[i] for i in train_loader.dataset.indices]
    val_labels = [mock_dataset.labels[i] for i in val_loader.dataset.indices]
    test_labels = [mock_dataset.labels[i] for i in test_loader.dataset.indices]

    assert len(train_labels) == pytest.approx(70, abs=1)
    assert len(val_labels) == pytest.approx(15, abs=1)
    assert len(test_labels) == pytest.approx(15, abs=1)

    assert train_labels.count(0) == pytest.approx(35, abs=1)
    assert train_labels.count(1) == pytest.approx(35, abs=1)

    assert val_labels.count(0) == pytest.approx(8, abs=1)
    assert val_labels.count(1) == pytest.approx(8, abs=1)

    assert test_labels.count(0) == pytest.approx(8, abs=1)
    assert test_labels.count(1) == pytest.approx(8, abs=1)

@patch("src.ml.preprocessing.dataloader.DocumentDataset")
def test_dataloader_batch_size(mock_image_dataset, mock_dataset):
    mock_image_dataset.return_value = mock_dataset

    train_loader, _, _ = get_image_dataloader(
        root_dir="dummy_path",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=10,
        shuffle=True
    )

    for batch in train_loader:
        images, labels = batch
        assert len(images) == pytest.approx(10, abs=1)
        assert len(labels) == pytest.approx(10, abs=1)

