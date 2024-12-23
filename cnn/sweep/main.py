import os.path
import torch
import wandb
from src.net.cnn import Net
from src.data.MergedDataset import MergedDataset, to_device
from utils import save_model, load_model, train, test
from sweep.config import sweep_confiuration


def exec(device):

    with wandb.init(config=sweep_confiuration):
        config = wandb.config

        if config.dataset == 'CXr':
            dataset = MergedDataset(device, chest_xray=True, nih_xray8=False, kaggle_rsna=False, batch_size=config.batch_size)
        elif config.dataset == 'NIH':
            dataset = MergedDataset(device, chest_xray=False, nih_xray8=True, kaggle_rsna=False, batch_size=config.batch_size, nihXray_drop_normal_percentage=0.97)
        elif config.dataset == 'RSNA':
            dataset = MergedDataset(device, chest_xray=False, nih_xray8=False, kaggle_rsna=True, batch_size=config.batch_size, kaggleRsna_drop_normal_percentage=0.50, split_seed=2024)
        elif config.dataset == 'CXr+NIH':
            dataset = MergedDataset(device, chest_xray=True, nih_xray8=True, kaggle_rsna=False, batch_size=config.batch_size, nihXray_drop_normal_percentage=0.95, split_seed=2024)
        else:
            return None

        train_dl, test_dl1 = dataset.getDataLoader()
        train_class_count = dataset.getTrainClasses()
        weight = [train_class_count[1]/(train_class_count[0]+train_class_count[1]), train_class_count[0]/(train_class_count[0]+train_class_count[1])]

        path = f'net/models/model_{config.dataset}_{config.batch_size}_{config.dropout}_{config.layer_size}_{config.conv_size}.pth'
        if os.path.exists(path) and os.path.isfile(path):
            # Load model
            net = load_model(path, config.conv_size, config.layer_size, config.dropout)
            net = to_device(net, device)
        else:
            # Create model
            net = to_device(Net(config.conv_size, config.layer_size, config.dropout), device)
            # Train model
            train(wandb, net, device, train_dl, config.learning_rate, config.epochs, weight)
            save_model(path, net)

        # Load test dataset
        if config.dataset == 'CXr':
            testDataset2 = MergedDataset(device, chest_xray=False, nih_xray8=True, kaggle_rsna=False, train_percentage=0, nihXray_drop_normal_percentage=0.95, split_seed=2024)
            _, test_dl2 = testDataset2.getDataLoader()
            testDataset3 = MergedDataset(device, chest_xray=False, nih_xray8=False, kaggle_rsna=True, train_percentage=0, kaggleRsna_drop_normal_percentage=0.50, split_seed=2024)
            _, test_dl3 = testDataset3.getDataLoader()
        elif config.dataset == 'NIH':
            testDataset2 = MergedDataset(device, chest_xray=True, nih_xray8=False, kaggle_rsna=False, batch_size=config.batch_size)
            _, test_dl2 = testDataset2.getDataLoader()
            dataset3 = MergedDataset(device, chest_xray=False, nih_xray8=False, kaggle_rsna=True, train_percentage=0, kaggleRsna_drop_normal_percentage=0.50, split_seed=2024)
            _, test_dl3 = dataset3.getDataLoader()
        elif config.dataset == 'RSNA':
            testDataset1 = MergedDataset(device, chest_xray=True, nih_xray8=False, kaggle_rsna=False, batch_size=config.batch_size)
            _, test_dl1 = testDataset1.getDataLoader()
            dataset2 = MergedDataset(device, chest_xray=False, nih_xray8=True, kaggle_rsna=False, train_percentage=0, nihXray_drop_normal_percentage=0.95, split_seed=2024)
            _, test_dl2 = dataset2.getDataLoader()

        # Test model
        test(wandb, net, device, config.dataset, test_dl1)
        test(wandb, net, device, 'NIH' if config.dataset == 'CXr' else 'CXr', test_dl2)
        test(wandb, net, device, 'NIH' if config.dataset == 'RSNA' else 'RSNA', test_dl3)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    sweep_id = wandb.sweep(sweep_confiuration, project="fds-cnn")
    wandb.agent(sweep_id, function=lambda: exec(device), count=24)

