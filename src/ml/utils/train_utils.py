import torch
from src.utils.config import setup_logging
from src.ml.models import efficientnet, resnet, visiontransformer

_log = setup_logging()

def train_one_epoch(model: torch.nn, train_loader: torch.utils.data, optimizer: torch.optim, criterion: torch.nn, device: torch.device):
    model.to(device)
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    
    return total_loss / len(train_loader)

def validate(model, val_loader: torch.utils.data, criterion: torch.nn, device: torch.device):
    model.eval()
    total_loss, total, correct = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy


def get_model(model_config):
    """
    Create a model based on configuration.
    """
    model_type = model_config['architecture']
    
    if model_type == 'resnet':
        return resnet.ResNet(num_classes=model_config['num_classes'])
    elif model_type == 'efficientnet':
        return efficientnet.EfficientNetBase(num_classes=model_config['num_classes'])
    elif model_type == 'visiontransformer':
        return visiontransformer.VisionTransformer(num_classes=model_config['num_classes'], model_name=model_config['model_name'])
    else:
        _log.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")

def get_optimizer(model, optimizer_config):
    """
    Create an optimizer based on configuration.
    """
    optimizer_type = optimizer_config['type']
    lr = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)

    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        _log.error(f"Unsupported optimizer type: {optimizer_type}")
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def get_loss_function(loss_config):
    """
    Create a loss function based on configuration.
    """
    loss_type = loss_config['type']
    
    if loss_type == 'crossentropy':
        return torch.nn.CrossEntropyLoss()
    elif loss_type == 'mse':
        return torch.nn.MSELoss()
    else:
        _log.error(f"Unsupported loss type: {loss_type}")
        raise ValueError(f"Unsupported loss type: {loss_type}")

