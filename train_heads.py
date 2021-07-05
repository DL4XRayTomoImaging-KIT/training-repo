import torch
from torch.utils.data import DataLoader
import os
from catalyst.contrib.callbacks.wandb_logger import WandbLogger
from catalyst import dl

from torch.nn import BCEWithLogitsLoss
from tqdm.auto import tqdm
from src.datasets import EmbeddingDataset
from src.models import SegmentationModel, partial_load
from src.runners import EmbeddingRunner

ROOT = '/mnt/data/machine-learning/arina/mots_embeddings/'
prev_name = 'disjoint_full_data'

for file in tqdm(os.listdir(ROOT)):
    if file.endswith(".txt"):
        name = ''.join(file.split('.')[:-1])
        checkpoint_path = f'/home/ws/tb0536/arina_mots/training-repo/logs/logdir/{prev_name}/checkpoints/best.pth'
        cp = torch.load(checkpoint_path)
        model = SegmentationModel(seg_model_name='deeplabv3_resnet50',
                                  in_channels=1, out_channels=32,
                                  n_tasks=7, head_hidden=512)
        w = cp['model_state_dict']
        partial_load(model, w)

        with open(os.path.join(ROOT, file), 'r') as f:
            paths = list(map(str.strip, f.readlines()))

        train_paths, val_paths = [], []
        for path in paths:
            if path.split('/')[-1].startswith('train'):
                train_paths.append(os.path.join(ROOT, path))
            else:
                val_paths.append(os.path.join(ROOT, path))

        train_data, test_data = EmbeddingDataset(train_paths, int(file[0])), EmbeddingDataset(val_paths, int(file[0]))
        batch_size = 256
        loaders = {
            "train": DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True),
            "valid": DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True),
        }

        emb_runner = EmbeddingRunner()
        criterion = BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.heads[train_data.task_id].parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        n_epochs = 250
        # model training
        emb_runner.train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            loaders=loaders,
            logdir=f"/home/ws/tb0536/arina_mots/training-repo/logs/logdir/{name}",
            num_epochs=n_epochs,
            verbose=False,
            timeit=True,
            callbacks=[
                WandbLogger(project="unsupervised-segmentation", name=f'classification head {train_data.task_id}'),
                dl.CheckpointCallback(1),
                dl.EarlyStoppingCallback(10),
                dl.OptimizerCallback(),
                dl.SchedulerCallback(),
                dl.AUCCallback(output_key='probs'),
                dl.AccuracyCallback(output_key='preds')
            ]
        )

        prev_name = name