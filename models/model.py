from typing import Any

import lightning.pytorch as pl
from torch.optim import Adam, SGD, AdamW

from models.L2Loss import L2Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging as log
from models.unet_model import UNet
import matplotlib.pyplot as plt
import io
from PIL import Image



class AnimalClean(pl.LightningModule):
    def __init__(self, opts):
        super(AnimalClean, self).__init__()
        self.opts = opts
        self.model_opts = self.opts.model
        self.model = UNet(self.model_opts)
        self.criterion = L2Loss()
        self.save_hyperparameters()
        self.samples = None
        self.epoch = 0
        log.info(self.model)

    def get_optimizer(self, opts, parameters):
        training_opts = opts.training
        optimization_opts = opts.optimization
        learning_rate = optimization_opts.learning_rate

        optimizer = training_opts.optimizer
        log.info(f"Trying to initialize {optimizer} optimizer")

        if optimizer.lower() == "adam":
            return Adam(parameters, lr=learning_rate,
                        betas=(optimization_opts.adam.beta_1, optimization_opts.adam.beta_2))
        elif optimizer.lower() == "sgd":
            return SGD(parameters, lr=learning_rate, momentum=optimization_opts.sgd.momentum)
        elif optimizer.lower() == "adamw":
            return AdamW(parameters, lr=learning_rate,
                         betas=(optimization_opts.adam.beta_1, optimization_opts.adam.beta_2))
        else:
            raise Exception(f"Optimizer {optimizer} not recognized. \n"
                            f"Please either choose from 'Adam', 'AdamW', or 'SGF', or implement your own")

    def configure_optimizers(self) -> Any:
        optimizer = self.get_optimizer(self.opts, self.parameters())
        scheduler_opts = self.opts.scheduling
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      factor=scheduler_opts.factor,
                                      patience=scheduler_opts.patience,
                                      threshold=scheduler_opts.threshold)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": scheduler_opts.monitor
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        ground_truth = y["ground_truth"]
        output = self.model(x)
        loss = self.criterion(output, ground_truth)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self._log_metrics(output, y, "train")
        self._set_samples(x, ground_truth, output, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ground_truth = y["ground_truth"]
        output = self.model(x)
        loss = self.criterion(output, ground_truth)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self._log_metrics(output, y, "val")
        self._set_samples(x, ground_truth, output, "val")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        ground_truth = y["ground_truth"]
        output = self.model(x)
        loss = self.criterion(output, ground_truth)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self._log_metrics(output, y, "test")
        self._set_samples(x, ground_truth, output, "test")
        return loss

    def _convert_tensor_to_PIL(self, image_tensor, transpose=True):
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor[0]
        if transpose:
            image_tensor = image_tensor.T

        image_tensor = image_tensor.detach()
        image_tensor = image_tensor.cpu()
        image_tensor = image_tensor.numpy()

        fig, ax = plt.subplots(dpi=60)
        ax.imshow(image_tensor, origin="lower", interpolation=None)
        buffer = io.BytesIO()
        plt.savefig(buffer)
        buffer.seek(0)
        image = Image.open(buffer)
        plt.close(fig)
        return image

    def _log_samples(self, phase):
        samples = self.samples[phase]
        logger = self.logger
        if samples is None:
            log.info(f"Samples have not been set for phase {phase}")
            return

        if self.opts.monitoring.method == "wandb":
            i = []
            for img_idx in range(samples["input"].shape[0]):
                image_tensor = samples["input"][img_idx]
                image = self._convert_tensor_to_PIL(image_tensor)
                i.append(image)
            logger.log_image(f"{phase}/Input", i, self.epoch)

            i = []
            for img_idx in range(samples["output"].shape[0]):
                image_tensor = samples["output"][img_idx]
                image = self._convert_tensor_to_PIL(image_tensor)
                i.append(image)
            logger.log_image(f"{phase}/Denoised Output", i, self.epoch)

            i = []
            for img_idx in range(samples["ground_truth"].shape[0]):
                image_tensor = samples["ground_truth"][img_idx]
                image = self._convert_tensor_to_PIL(image_tensor)
                i.append(image)
            logger.log_image(f"{phase}/Ground Truth", i, self.epoch)

        else:
            logger = logger.experiment
            logger.add_images(f"{phase}/Input", samples["input"], self.epoch, dataformats="NCHW")
            logger.add_images(f"{phase}/Denoised Output", samples["output"], self.epoch, dataformats="NCHW")
            logger.add_images(f"{phase}/Ground Truth", samples["ground_truth"], self.epoch, dataformats="NCHW")

    def _set_samples(self, features, ground_truth, output_samples, phase):
        if self.samples is None or phase not in self.samples or "input" not in self.samples[phase]:
            self._initialize_samples()

        if self.samples[phase]["input"] is None:
            self.samples[phase]["input"] = features
        if self.samples[phase]["ground_truth"] is None:
            self.samples[phase]["ground_truth"] = ground_truth
        if self.samples[phase]["output"] is None:
            self.samples[phase]["output"] = output_samples

    def _initialize_samples(self):
        log.info(f"Initializing Sample dict")
        data = {}
        for phase in ["train", "val", "test"]:
            data[phase] = {
                "input": None,
                "output": None,
                "ground_truth": None
            }

        self.samples = data

    def on_train_epoch_start(self) -> None:
        self._initialize_samples()

    def on_train_epoch_end(self) -> None:
        self._log_samples("train")

    def on_test_epoch_end(self) -> None:
        self._log_samples("test")

    def on_validation_epoch_end(self) -> None:
        self._log_samples("val")
        self.epoch += 1

