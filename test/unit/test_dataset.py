import pytest
from src.ml.preprocessing.dataset import DocumentDataset
import torch
from torchvision import transforms
import os
from PIL import Image
from pathlib import Path

@pytest.fixture
def mock_image_dataset(tmp_path):
    classes = ["class1", "class2"]
    root_dir = Path(tmp_path) / "temp_dataset"
    root_dir.mkdir()

    for class_name in classes:
        class_dir = Path(root_dir) / class_name
        class_dir.mkdir()

        for i in range(5):
            img = Image.new("RGB", (128, 128),color=(i*10, i*10, i*10))
            img.save(os.path.join(class_dir, f"{i}.jpg"))
    
    return root_dir

def test_dataset_length(mock_image_dataset):
    dataset = DocumentDataset(mock_image_dataset)
    assert len(dataset) == 10

def test_dataset_getitem(mock_image_dataset):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = DocumentDataset(mock_image_dataset, transform=transform)
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
    assert image.shape == (3, 128, 128)

def test_dataset_labels(mock_image_dataset):
    dataset = DocumentDataset(mock_image_dataset)
    labels = [dataset[i][1] for i in range(len(dataset))]

    assert all(label in [0,1] for label in labels)

def test_dataset_transform(mock_image_dataset):
    transformation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = DocumentDataset(mock_image_dataset, transform=transformation)
    image, _ = dataset[0]

    assert torch.min(image) >= -1.0
    assert torch.max(image) <= 1.0