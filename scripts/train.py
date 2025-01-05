import torch
import mlflow
import mlflow.pytorch
from src.ml.utils.train_utils import train_one_epoch, validate, get_optimizer, get_model, get_loss_function
from src.ml.preprocessing.dataloader import get_image_dataloader
from torchvision import transforms
from src.utils.config import load_config, setup_logging
from tqdm import tqdm
from src.ml.utils.mlflow_utils import log_mlflow_metrics, log_mlflow_params
from PIL import Image, ImageOps
_log = setup_logging()
config = load_config()

def main():
    try:
        model_config = config['model']
        _log.info("Model configuration loaded successfully")
    except KeyError:
        _log.error("Model configuration not found in config file")
        raise

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    _log.info("Dataset Transformation")
    
    train_loader, val_loader, test_loader = get_image_dataloader(
        root_dir=model_config['dataset_path'],
        train_ratio=model_config['train_ratio'],
        val_ratio=model_config['val_ratio'],
        test_ratio=model_config['test_ratio'],
        batch_size=model_config['batch_size'],
        transform=transform
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log.info(f"Using device: {device}")

    model = get_model(model_config)
    model.to(device)

    optimizer = get_optimizer(model, model_config['optimizer'])
    criterion = get_loss_function(model_config['criterion'])

    log_mlflow_params(model_config)

    progress_bar = tqdm(range(1, model_config['num_epochs'] + 1), desc="Training", position=0, leave=True)
    for epoch in progress_bar:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        log_mlflow_metrics(epoch, {'train_loss': train_loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

        progress_bar.set_description(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    _log.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    model_name = model_config['model_name']
    epoch_str = f"epoch_{epoch}"
    model_save_path = f"models/{model_name}_{epoch_str}"
    mlflow.pytorch.log_model(model, model_save_path)
    _log.info(f"Model saved at: {model_save_path}")

if __name__ == "__main__":
    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run():
            main()
    except Exception as e:
        _log.error(f"Error occurred: {e}")
        raise