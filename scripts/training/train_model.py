import sys
import os
import yaml
import glob
import argparse
import shutil
import socket
import wandb
from datetime import datetime

from pytorch_lightning.loggers import CometLogger

import torch
import pytorch_lightning
from torchsummary import summary

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import TQDMProgressBar as ProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.profilers import AdvancedProfiler

from lightning import Set2TreeLightning

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    """
    Argument parser for training script.
    """
    parser = argparse.ArgumentParser(description="Train the GNN tagger.")

    parser.add_argument("-c", "--config", required=True, type=str, help="Path to config file.")
    parser.add_argument("--gpus", default="", type=str, help="Comma separated list of GPUs to use.")
    parser.add_argument("--ckpt_path", type=str, help="Restart training from a checkpoint.")
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="No logging, checkpointing, test on a few jets.",
    )
    parser.add_argument("--batch_size", type=int, help="Overwrite config batch size.")
    parser.add_argument("--reduce_dataset", type=float, help="Modify dataset size on the fly.")
    parser.add_argument("--num_workers", type=int, help="Overwrite config num workers.")
    parser.add_argument("--num_epochs", type=int, help="Overwrite config num epochs.")
    parser.add_argument("--no_logging", action="store_true", help="Disable the logging framework.")
    # parser.add_argument("--adding_noise", action="store_true", default=False, help="Adding noise to features.")
    # parser.add_argument("--mean", default=0.0, type=float, help="Mean value of noise")
    # parser.add_argument("--std", default=1.0, type=float, help="Amplification factor of noise standard deviation")
    # parser.add_argument('--lambda_amp', type=float, default=0.1, help="amp parameter")
    # parser.add_argument('--normalize_features', action="store_true", default=False)
    # parser.add_argument('--step_size', type=float, default=0.001, help="Degree of perturbation at each step")
    # parser.add_argument('--m', type=int, default=3, help="Number of rounds of training perturbations per batch")
    # parser.add_argument('--lambda', type=float, default=0.1, help="Hyperparameters weighted to loss")
    parser.add_argument("--perturb_ratio", default=0.5, type=float, help="Percentage of leaf nodes with perturb")
    parser.add_argument("--perturb_method", default='gaussian', type=str, help="method of perturb")


    args = parser.parse_args()
    return args


def update_config(args, config):
    """
    Update config with passed arguments
    """
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    return config


def setup_logger(args, config):
    """
    Handle logging for the DDP multi-gpu training.
    I think this is really a bug on comet's end that they don't
    automatically group the experiments...
    """

    # no logging
    if args.test_run or args.no_logging:
        return None

    # need to set up a new experiment
    comet_logger = setup_comet_logger(config, args, os.environ.get("COMET_EXP_ID"))
    if comet_logger.experiment.get_key():
        os.environ["COMET_EXP_ID"] = comet_logger.experiment.get_key()

    return comet_logger


def setup_comet_logger(config, args, exp_id=None):

    # initialise logger
    comet_logger = CometLogger(
        api_key=comet_api_key,
        save_dir="logs",
        project_name=comet_project_name,
        workspace=comet_workspace,
        experiment_name=config["name"],
        experiment_key=exp_id,
    )

    # log config, hyperparameters and source files
    if os.environ.get("LOCAL_RANK") is None:
        # comet_logger.experiment.log_parameter('sample',            config['sample'])
        # comet_logger.experiment.log_parameter('info',              config['info'])
        # comet_logger.experiment.log_parameter('use_leptons',       config['use_leptons'])
        # comet_logger.experiment.log_parameter('batch_size',        config['batch_size'])
        # comet_logger.experiment.log_parameter('learning_rate',     config['learning_rate'])
        # comet_logger.experiment.log_parameter('reduce_dataset',    config['reduce_dataset'])
        # comet_logger.experiment.log_parameter('num_epochs',        config['num_epochs'])
        # comet_logger.experiment.log_parameter('num_gpus',          config['num_gpus'])
        # comet_logger.experiment.log_parameter('num_workers',       config['num_workers'])
        # comet_logger.experiment.log_parameter('move_files_temp',   config['move_files_temp'])
        # comet_logger.experiment.log_parameter('use_swa',           config['use_swa'])

        comet_logger.experiment.log_parameter("torch_version", torch.__version__)
        comet_logger.experiment.log_parameter(
            "lightning_version", pytorch_lightning.__version__
        )
        comet_logger.experiment.log_parameter("cuda_version", torch.version.cuda)
        comet_logger.experiment.log_parameter("hostname", socket.gethostname())

        comet_logger.experiment.log_asset(args.config)
        # comet_logger.experiment.log_asset(config['scale_dict'])
        # comet_logger.experiment.log_asset(config['var_config'])
        # all_files = glob.glob('./*.py') + glob.glob('models/*.py')
        # for fpath in all_files:
        #     comet_logger.experiment.log_code(fpath)

    return comet_logger


def get_callbacks(config, args):
    """
    Initialise training callbacks
    """

    refresh_rate = 1 if args.test_run else 20
    callbacks = [ProgressBar(refresh_rate=refresh_rate)]

    # initialise checkpoint callback
    if not args.test_run:
        # if 'val_jet_loss' in config['logging_losses']:
        #     monitor_loss = 'val_jet_loss'
        # else:
        monitor_loss = "val_loss"

        # filename template
        file_name = config["run_name"] + "-{epoch:02d}-{" + monitor_loss + ":.4f}"

        # callback
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_loss,
            dirpath=os.path.join("saved_models/", config["run_name"], "ckpts"),
            filename=file_name,
            save_top_k=-1,
        )
        callbacks += [checkpoint_callback]

    if config["use_swa"]:
        callbacks += [
            StochasticWeightAveraging(
                swa_lrs=1e-4, swa_epoch_start=0.7, annealing_epochs=15
            )
        ]

    return callbacks


def recurse_model(module, params, submodule_name=None):
    for name, submodule in module._modules.items():
        name = submodule_name if submodule_name else name
        if hasattr(submodule, "weight") and hasattr(submodule.weight, "size"):
            params[name] += torch.prod(
                torch.LongTensor(list(submodule.weight.size()))
            ).item()
        if hasattr(submodule, "bias") and hasattr(submodule.bias, "size"):
            params[name] += torch.prod(
                torch.LongTensor(list(submodule.bias.size()))
            ).item()
        if hasattr(submodule, "_modules"):
            recurse_model(submodule, params, name)
# def summary(model):
#     params = {name: 0 for name in model._modules.keys()}
#     recurse_model(model, params)
#     print("-" * 100)
#     print("Detailed model summary (approximate):")
#     print("-" * 100)
#     for module, n in params.items():
#         print(f"{module:.<20}{n:.>10,d}")
#     print("-" * 100)
#     print(f'{"Total params":.<20}{sum(params.values()):.>10,d}')
#     print("-" * 100)
#     print()

def train(args, config, logger, config_path):
    """
    Fit the model.
    """

    # create a new model
    model = Set2TreeLightning(config_path)

    # model summary
    print("summary begin:")
    print(model.model)
    print("summary end\n")

    # log number of parametesr
    if os.environ.get("LOCAL_RANK") is None and logger is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.experiment.log_parameter("trainable_params", trainable_params)

    # if we pass a checkpoint file, load the previous network
    if args.ckpt_path:
        print("Loading previously trained model from checkpoint file:", args.ckpt_path)
        model = Set2TreeLightning.load_from_checkpoint(checkpoint_path=args.ckpt_path,cfg_path=args.config)

    # share workers between GPUs
    if config["num_gpus"]:
        config["num_workers"] = config["num_workers"] // config["num_gpus"]
    config["run_name"] = config["run_name"] + datetime.now().strftime("_%Y-%m-%d_%H:%M")
    path = os.path.join("saved_models/", config["run_name"])
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + "/config.yaml", "w") as f:
        yaml.dump(config, f)
    # get callbacks
    callbacks = get_callbacks(config, args)
    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    # create the lightening trainer
    print("Creating trainer...")
    trainer = Trainer(
        max_epochs=config["num_epochs"],
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator=config["accelerator"],
        devices=config["num_gpus"],
        logger=logger,
        precision=32,
        log_every_n_steps=20,
        # fast_dev_run=args.test_run,
        callbacks=callbacks,
        # limit_val_batches=0  #validation_step before train
        # limit_test_batches=5000,
        # limit_train_batches=5000,
        # limit_val_batches=5000,
        # profiler=profiler
    )
    # get the the graph datamodule
    # datamodule = du.get_datamodule(config, logger)

    # fit model model
    print("Fitting model...")
    trainer.fit(model)

    return model, trainer


def print_job_info(args, config):
    """
    Print job information.
    """

    if os.environ.get("LOCAL_RANK") is not None:
        return

    print("-" * 100)
    print("torch", torch.__version__)
    print("lightning", pytorch_lightning.__version__)
    print("cuda", torch.version.cuda)
    import dgl

    print("dgl", dgl.__version__)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("Visible GPUs:", args.gpus, " - ", device_name)
    print("-" * 100, "\n")


def parse_gpus(config, gpus):

    # set available GPUs based on arguments
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    num_gpus = len(gpus.split(",")) if gpus != "" else None
    accelerator = "gpu" if num_gpus is not None else "cpu"
    config["accelerator"] = accelerator
    config["num_gpus"] = num_gpus

    return config


def cleanup(config, model):

    print("-" * 100)
    print("Cleaning up...")
    # keep main process only
    if model.global_rank != 0:
        sys.exit(0)

    # fu.remove_files_temp(config, tag="Training")


def main():
    """
    Training entry point.
    """
    # pytorch_lightning.seed_everything(42)

    # parse args
    args = parse_args()

    # read config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # overwrite config using args
    config = update_config(args, config)

    # parse the gpu argument and update config
    config = parse_gpus(config, args.gpus)

    # print job info once
    print_job_info(args, config)

    # setup logger
    # logger = setup_logger(args, config)
    logger=None

    # login wandb
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="training NRI with noise(10 percent,0.1 std)",
        # Track hyperparameters and run metadata
        config={
            "run_name": config["run_name"],
            # "adding_noise": config["adding_noise"],
            # "noise_ratio": config["noise_ratio"],
            # "std": config["std"],
        })
    # copy files to output dir for reproducability
    # if not args.test_run:
    #     config = fu.prep_out_dir(args, config)

    # Move files to temp directory
    # if not args.test_run:
    #     config = fu.move_files_temp(config, tag="Training")

    # run training
    model, trainer = train(args, config, logger, args.config)

    # cleanup
    if not args.test_run:
        cleanup(config, model)


if __name__ == "__main__":
    main()
