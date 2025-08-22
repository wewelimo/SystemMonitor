import glob
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary

from core.datasets.datasetLoader import get_dataloaders
from core.networks.model import Model

# torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    ##########################################################
    checkpoint = "./lightning_logs/version_11/checkpoints"
    dataset_root = "dataset/"

    ##########################################################
    checkpoint = glob.glob(os.path.join(checkpoint, "*.ckpt"))[-1]

    # Dataset
    dataloader_train, dataloader_val = get_dataloaders(dataset_root, num_workers=8, batch_size=32)

    # Model
    # model = Model(lr=1e-4)
    model = Model.load_from_checkpoint(checkpoint, strict=True, lr=1e-5)

    # Print model summary
    summary = ModelSummary(model, max_depth=4)
    print(summary)

    # Compile the model
    # model = torch.compile(model)

    valid = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    valid.validate(model=model, dataloaders=dataloader_val)

    # Training
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="loss/val")
    checkpoint_callback2 = ModelCheckpoint(save_top_k=1, monitor="Acc/val_mean")
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=500, num_sanity_val_steps=0, precision="16-mixed", callbacks=[checkpoint_callback, checkpoint_callback2])
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # tensorboard --logdir=lightning_logs/
