# from pytorch_lightning.loggers import CometLogger
# from pytorch_lightning import Trainer
from pathlib import Path

import dgl
import json
import numpy as np
import torch
import wandb
import copy
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader

from set2tree.config import load_config
from set2tree.data import PhasespaceSet, create_dataloader_mode_tags
from set2tree.losses import EMDLoss, FocalLoss, SupGraphConLoss
from set2tree.models.NRI import NRIModel
from set2tree.models.SGCL import SGCL

from set2tree.utils import calculate_class_weights
from pl_metrics import *


class Set2TreeLightning(LightningModule):
    def __init__(self, cfg_path):
        super().__init__()

        config, tags = load_config(Path(cfg_path).resolve())
        self.config = config
        if config["train"]["model"] == "nri_model":
            self.model = NRIModel(**self.config["model"])
        elif config["train"]["model"] == "pascl":
            self.model = SGCL(**self.config["model"])



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
        self.adv_sup_con_loss = SupGraphConLoss(
            temperature=0.07, contrast_mode='all',base_temperature=0.07)
        self.train_accuracy = PerfectLCAG(
            batch_size=config["train"]["batch_size"], ignore_index=-1, ).to(device)
        self.val_accuracy = PerfectLCAG(
            batch_size=config["val"]["batch_size"], ignore_index=-1).to(device)
    def forward(self, g):

        return self.model(g)

    def log_losses(self, loss, batch_size, stage):
        self.log(f"{stage}_loss", loss["loss"], sync_dist=True, batch_size=batch_size)
        for t, l in loss.items():
            n = f"{stage}_{t}_loss" if "loss" not in t else f"{stage}_{t}"
            self.log(n, l, sync_dist=True, batch_size=batch_size)

    def shared_step(self, batch, evaluation=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out = self.model(batch.to(device))

        if evaluation:
            print("shared_step evaluation")
            return out.edata["pred"], out.edata["lca"], out.batch_num_edges(), out.batch_num_nodes(), None

        # compute total loss
        # loss["loss"] = sum(subloss for subloss in loss.values())

        # Supervised Contrastive Learning with Perturbative Augmentation(PASCL)
        if self.config["train"]["model"] == "pascl":
            loss = {}
            if self.training:
                ### feature adversarial perturbation
                if self.config["perturb_method"] == "adversarial":
                    forward = lambda perturb: self.model(batch, perturb).to(device)
                    model_forward = (self.model, forward)
                    y = batch.edata["lca"]
                    perturb_shape = (batch.num_nodes(), 4)
                    aug_out = self.flag(model_forward, perturb_shape, y, self.config, self.configure_optimizers(), self.loss_fn)

                ### Gaussian noise perturbation
                elif self.config["perturb_method"] == "gaussian":
                    std = torch.std(batch.ndata["leaf features"], dim=1) * self.config["std"]
                    num_nodes = batch.num_nodes()
                    num_perturb = int(num_nodes * self.config["perturb_ratio"])
                    perturb_idx = torch.randperm(num_nodes)[:num_perturb]
                    for idx in perturb_idx:
                        noise = torch.normal(mean = 0.0, std = std[idx], size = batch.ndata["leaf features"][idx].shape).to(device)
                        batch.ndata["leaf features"][idx] += noise
                    aug_out = self.model(batch.to(device))

                ### node mask
                elif self.config["perturb_method"] == "mask":
                    num_nodes = batch.num_nodes()
                    num_perturb = int(num_nodes * self.config["perturb_ratio"])
                    perturb_idx = torch.randperm(num_nodes)[:num_perturb]
                    for idx in perturb_idx:
                        batch.ndata["leaf features"][idx].zero_()
                    aug_out = self.model(batch.to(device))

                loss["loss"] = self.config["lambda"] * self.adv_sup_con_loss(out,aug_out) + \
                               (1 - self.config["lambda"]) * (self.loss_fn(out.edata["pred"], out.edata["lca"]) + self.loss_fn(aug_out.edata["pred"], aug_out.edata["lca"]))
            else:
                out = self.model(batch.to(device))
                loss["loss"] = self.loss_fn(out.edata["pred"], out.edata["lca"])


        # NRI and others
        elif self.config["train"]["model"] in ["nri_model"]:
            loss = {}
            ### Adding perturbations to nodes as data augmentation on NRI model
             # if self.training:
            #     print(self.config["std"],self.config["perturb_ratio"])
            #     std = torch.std(batch.ndata["leaf features"], dim=1) * self.config["std"]
            #     num_nodes = batch.num_nodes()
            #     num_perturb = int(num_nodes * self.config["perturb_ratio"])
            #     perturb_idx = torch.randperm(num_nodes)[:num_perturb]
            #     for idx in perturb_idx:
            #         noise = torch.normal(mean = 0.0, std = std[idx], size = batch.ndata["leaf features"][idx].shape).to(device)
            #         print(noise)
            #         batch.ndata["leaf features"][idx] += noise
            #
            #     aug_out = self.model(batch.to(device))
            #
            #     loss["loss"] = self.loss_fn(aug_out.edata["pred"], aug_out.edata["lca"])
            #     return aug_out.edata["pred"], aug_out.edata["lca"], aug_out.batch_num_edges(), aug_out.batch_num_nodes(), loss
            # else:
            #     loss["loss"] = self.loss_fn(out.edata["pred"], out.edata["lca"])
            # print('nri_model')
            loss["loss"] = self.loss_fn(out.edata["pred"], out.edata["lca"])

        else:
            raise ValueError(f"no such model")
        return out.edata["pred"], out.edata["lca"], out.batch_num_edges(), out.batch_num_nodes(), loss



    def flag(self, model_forward, perturb_shape, y, args, optimizer, criterion):
        ### implementation of paper: Robust Optimization as Data Augmentation for Large-scale Graphs, Accepted at CVPR 2022.
        ### code: implementation at https://github.com/devnkong/FLAG.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model, forward = model_forward
        model.train()
        optimizer.zero_grad()

        # perturb=torch.nn.Parameter(torch.ones(perturb_shape)).to(device)
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-args["step_size"], args["step_size"]).to(device)
        perturb.requires_grad_(True)

        aug_out = forward(perturb).to(device)
        loss = criterion(aug_out.edata["pred"], y)
        loss /= args["m"]


        for _ in range(args["m"] - 1):
            loss.backward()
            # print("grad:",perturb,perturb.grad)
            perturb_data = perturb.detach() + args["step_size"] * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            aug_out = forward(perturb)
            loss = criterion(aug_out.edata["pred"], y)
            loss /= args["m"]
        return aug_out


    def training_step(self, batch, batch_idx):
        preds, labels, edge_batch, node_batch, loss = self.shared_step(batch)
        # print(batch,batch_idx)
        # log losses
        self.log_losses(loss, batch_size=self.config["train"]["batch_size"], stage="train")

        # wandb log
        # if batch_idx % 20 == 0:
        #     acc=self.train_accuracy((preds, labels, edge_batch, node_batch))
        #     wandb.log({"train_acc": acc, "loss": loss})
        # print("training_step:",loss,loss["loss"])
        return loss["loss"]



    def validation_step(self, batch, batch_idx):
        preds, labels, edge_batch, node_batch, loss = self.shared_step(batch)

        # log losses
        self.log_losses(loss, batch_size=self.config["val"]["batch_size"], stage="val")

        # wandb log
        # acc=self.val_accuracy((preds, labels, edge_batch, node_batch))
        # wandb.log({"val_acc": acc, "loss": loss})

        # return loss (and maybe more stuff)
        # return_dict = loss
        #
        # return return_dict
        return loss["loss"]


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
            # noise_ratio=self.config["noise_ratio"],
            mode="train",
            # adding_noise = self.config["adding_noise"],
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
        print("self.config[train][batch_size]:",self.config["train"]["batch_size"])
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
            # noise_ratio=self.config["noise_ratio"],
            mode="val",
            # adding_noise=self.config["adding_noise"],
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
