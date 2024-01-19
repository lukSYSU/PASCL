# from pytorch_lightning.loggers import CometLogger
# from pytorch_lightning import Trainer
from pathlib import Path

import dgl
import json
import numpy as np
import torch
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader

from set2tree.config import load_config
from set2tree.data import PhasespaceSet, create_dataloader_mode_tags
from set2tree.losses import EMDLoss, FocalLoss
from set2tree.models.NRI import NRIModel
from set2tree.models.Set2TreeGAT import Set2TreeGAT
from set2tree.models.Set2TreeEdge import Set2TreeEdge
from set2tree.utils import calculate_class_weights


class Set2TreeLightning(LightningModule):
    def __init__(self, cfg_path):
        super().__init__()

        config, tags = load_config(Path(cfg_path).resolve())
        self.config = config
        if config["train"]["model"] == "nri_model":
            self.model = NRIModel(**self.config["model"])
        elif config["train"]["model"] == "gat_model":
            self.model = Set2TreeGAT(**self.config["model"])
        elif config["train"]["model"] == "edge_model":
            self.model = Set2TreeEdge(**self.config["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = None
        mode_tags = create_dataloader_mode_tags(self.config, tags)
        if self.config["train"]["class_weights"]:
            class_weights = calculate_class_weights(
                dataloader=mode_tags["Training"][2],
                num_classes=self.config["dataset"]["num_classes"],
                num_batches=100,
                amp_enabled=self.config["train"]["mixed_precision"],
            )
            print(class_weights)
            class_weights = class_weights.to(device)
        if self.config["loss"] == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights, ignore_index=-1, reduction="mean"
            )
        elif self.config["loss"] == "focal":
            self.loss_fn = FocalLoss(
                gamma=2.5, ignore_index=-1, weight=class_weights, reduction="mean"
            )

    def forward(self, g):

        return self.model(g)

    def log_losses(self, loss, batch_size, stage):
        self.log(f"{stage}_loss", loss["loss"], sync_dist=True, batch_size=batch_size)
        for t, l in loss.items():
            n = f"{stage}_{t}_loss" if "loss" not in t else f"{stage}_{t}"
            self.log(n, l, sync_dist=True, batch_size=batch_size)

    def shared_step(self, batch, evaluation=False):
        out = self.model(batch)

        if evaluation:
            return out.edata["pred"], out.edata["lca"], out.batch_num_nodes(), None

        # compute total loss
        # loss["loss"] = sum(subloss for subloss in loss.values())
        loss = {}
        loss["loss"] = self.loss_fn(out.edata["pred"], out.edata["lca"])

        return out.edata["pred"], out.edata["lca"], out.batch_num_nodes(), loss

    def training_step(self, batch, batch_idx):
        # foward pass
        preds, labels, num_nodes, loss = self.shared_step(batch)

        # log losses
        self.log_losses(
            loss, batch_size=self.config["train"]["batch_size"], stage="train"
        )

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        # foward pass
        preds, labels, num_nodes, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, batch_size=self.config["val"]["batch_size"], stage="val")

        # return loss (and maybe more stuff)
        return_dict = loss

        return return_dict

    def configure_optimizers(self):
        if self.config["train"]["optimiser"] == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                self.config["train"]["learning_rate"],
                amsgrad=False,
            )
        elif self.config["train"]["optimiser"] == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(), self.config["train"]["learning_rate"]
            )
        elif self.config["train"]["optimiser"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.model.parameters(), self.config["train"]["learning_rate"]
            )
        return optimizer

    def train_dataloader(self):

        dataset = PhasespaceSet(
            root=self.config["train_path"],
            mode="train",
            file_ids=self.config["dataset"]["datasets"]
            if "datasets" in self.config["dataset"]["phasespace"]
            else None,
            **self.config["dataset"]["phasespace"]["config"],
        )

        # If scaling is requested and there's no scaling factors in the configs, extract them from the dataset
        # They will be calculated by the first dataset created without a scaling_dict given
        if (
            self.config["dataset"]["phasespace"]["config"]["apply_scaling"]
            and self.config["dataset"]["phasespace"]["config"]["scaling_dict"] is None
        ):
            self.config["dataset"]["phasespace"]["config"][
                "scaling_dict"
            ] = dataset.scaling_dict
        print(
            f"{type(dataset).__name__} created for training with {dataset.__len__()} samples"
        )
        # with open('scaling_dict_train.json', 'w') as f:
        #     json.dump(dataset.scaling_dict, f)
        print(dataset.scaling_dict)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config["train"]["batch_size"],
            drop_last=True,
            shuffle=True,
            # collate_fn=collate_fn,
            num_workers=self.config["train"]["num_workers"],
            pin_memory=False,
            prefetch_factor=self.config["train"]["batch_size"],
            collate_fn=PhasespaceSet.collate_graphs,
        )
        return dataloader

    def val_dataloader(self):
        dataset = PhasespaceSet(
            root=self.config["val_path"],
            mode="val",
            file_ids=self.config["dataset"]["datasets"]
            if "datasets" in self.config["dataset"]["phasespace"]
            else None,
            **self.config["dataset"]["phasespace"]["config"],
        )

        # If scaling is requested and there's no scaling factors in the configs, extract them from the dataset
        # They will be calculated by the first dataset created without a scaling_dict given
        if (
            self.config["dataset"]["phasespace"]["config"]["apply_scaling"]
            and self.config["dataset"]["phasespace"]["config"]["scaling_dict"] is None
        ):
            self.config["dataset"]["phasespace"]["config"][
                "scaling_dict"
            ] = dataset.scaling_dict
        print(
            f"{type(dataset).__name__} created for validation with {dataset.__len__()} samples"
        )
        print(dataset.scaling_dict)
        # with open('scaling_dict_val.json', 'w') as f:
        #     json.dump(dataset.scaling_dict, f)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config["val"]["batch_size"],
            drop_last=True,
            shuffle=False,
            # collate_fn=collate_fn,
            num_workers=self.config["val"]["num_workers"],
            pin_memory=False,
            prefetch_factor=self.config["val"]["batch_size"],
            collate_fn=PhasespaceSet.collate_graphs,
        )

        return dataloader
