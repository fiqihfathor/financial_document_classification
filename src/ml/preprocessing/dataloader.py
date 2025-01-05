from typing import Optional, Callable, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from src.ml.preprocessing.dataset import DocumentDataset

def get_image_dataloader(
        root_dir: str,
        train_ratio: float=0.7,
        val_ratio: float=0.15,
        test_ratio: float=0.15,
        batch_size: int=32,
        shuffle: bool=True,
        transform: Optional[Callable]=None
    )-> Tuple[DataLoader, DataLoader, DataLoader]: 
    dataset = DocumentDataset(root_dir, transform=transform)
    labels = [dataset[i][1] for i in range(len(dataset))]
    
    train_idx, val_test_idx, _, _ = train_test_split(
        range(len(dataset)),
        labels,
        stratify=labels,
        test_size=1 - train_ratio,
        random_state=42
    )
    
    val_idx, test_idx = train_test_split(
        val_test_idx,
        stratify=[labels[i] for i in val_test_idx],
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42
    )
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    batch_size = min(batch_size, len(train_dataset), len(val_dataset), len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, val_loader, test_loader