#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import multiprocessing as mtp
import wandb
import datetime

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
#from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from detect import _draw_and_save_output_image_online
from test import _evaluate, _create_validation_data_loader
from distutils.util import strtobool

from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False, overfitting=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not overfitting,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


# def run():
print_environment_info()
parser = argparse.ArgumentParser(description="Trains the YOLO model.")
parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
parser.add_argument("-d", "--data", type=str, default="config/crowdhuman-416x416.data", help="Path to data config file (.data)")
parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
parser.add_argument("--n_cpu", type=int, default=min(mtp.cpu_count(), 8), help="Number of cpu threads to use during batch generation")
parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
parser.add_argument("--overfitting", action="store_true", help="Overfitting test")
parser.add_argument("--collect_data", type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
parser.add_argument("--hyperbolic", type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
args = parser.parse_args()
print(f"Command line arguments: {args}")

if args.seed != -1:
    provide_determinism(args.seed)

# Create output directories if missing
os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Get data configuration
data_config = parse_data_config(args.data)
train_path = data_config["train"]
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ############
# Create model
# ############

model = load_model(args.model, args.pretrained_weights)

# Print model
if args.verbose:
    summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions'] if not args.overfitting else 1
if args.overfitting: 
    args.epochs = 10
    model.hyperparams["decay"] = 0

##################
# Initialize wandb
##################
wandb_config = dict(
    Name="COMS6998-Representation Learning",
    algorithm=f"{'-'.join(str(datetime.datetime.now())[5:16].split())}-Yolo-v3-hyperbolic-{args.hyperbolic}",
    overfitting=args.overfitting,
    **model.hyperparams,
)
if args.collect_data:
    wandb.init(project=wandb_config['Name'], entity='jiayinsen', config=wandb_config, name=wandb_config['algorithm'])
else: wandb.init(mode="disabled")

# #################
# Create Dataloader
# #################

# Load training dataloader
dataloader = _create_data_loader(
    train_path,
    mini_batch_size,
    model.hyperparams['height'],
    args.n_cpu,
    args.multiscale_training,
    overfitting=args.overfitting)

# Load validation dataloader
validation_dataloader = _create_validation_data_loader(
    valid_path,
    mini_batch_size,
    model.hyperparams['height'],
    args.n_cpu)

# ################
# Create optimizer
# ################

params = [p for p in model.parameters() if p.requires_grad]

if (model.hyperparams['optimizer'] in [None, "adam"]):
    optimizer = optim.Adam(
        params,
        lr=model.hyperparams['learning_rate'],
        weight_decay=model.hyperparams['decay'],
    )
elif (model.hyperparams['optimizer'] == "sgd"):
    optimizer = optim.SGD(
        params,
        lr=model.hyperparams['learning_rate'],
        weight_decay=model.hyperparams['decay'],
        momentum=model.hyperparams['momentum'])
else:
    print("Unknown optimizer. Please choose between (adam, sgd).")

### Overfitting Test ###
overfitting_iterations = 1000; sample_index = 10
if args.overfitting:
    print("\n---- Overfitting Test ----")
    iter_dataloader = iter(dataloader)
    for i in range(sample_index):
        _, imgs, targets = next(iter_dataloader)
    imgs = imgs.to(device, non_blocking=True)
    print(f"Image Shape: {imgs.shape}")
    targets = targets.to(device)
    
    for epoch in range(1, args.epochs+1):
        model.eval() # why eval has effect on output???
        with torch.no_grad():
            outputs = model(imgs)
            _draw_and_save_output_image_online(imgs.squeeze(0), outputs, img_size=imgs.shape[-1], 
                                                output_path=f"output/overfitting{epoch}.jpg", classes=class_names)
        model.train()
        for batch_i, _ in enumerate(tqdm.tqdm(range(overfitting_iterations), desc=f"Overfitting Epoch: {epoch}")):
            batches_done = overfitting_iterations * epoch + batch_i
            outputs = model(imgs)
            loss, loss_components = compute_loss(outputs, targets, model)
            loss.backward()
            ###############
            # Run optimizer
            ###############
            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                wandb.log({"steps":batches_done, 'train/learning_rate': lr})
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)
            wandb.log({"steps": batches_done, "train/iou_loss": float(loss_components[0])})
            wandb.log({"steps": batches_done, "train/obj_loss": float(loss_components[1])})
            wandb.log({"steps": batches_done, "train/cls_loss": float(loss_components[2])})
            wandb.log({"steps": batches_done, "train/loss_all": to_cpu(loss).item()})

            model.seen += imgs.size(0)

else:
    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                wandb.log({"steps":batches_done, 'train/learning_rate': lr})
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging

            wandb.log({"steps": batches_done, "train/iou_loss": float(loss_components[0])})
            wandb.log({"steps": batches_done, "train/obj_loss": float(loss_components[1])})
            wandb.log({"steps": batches_done, "train/cls_loss": float(loss_components[2])})
            wandb.log({"steps": batches_done, "train/loss_all": to_cpu(loss).item()})

            model.seen += imgs.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                wandb.log({"epochs": epoch, "validation/precision": precision.mean()})
                wandb.log({"epochs": epoch, "validation/recall": recall.mean()})
                wandb.log({"epochs": epoch, "validation/mAP": AP.mean()})
                wandb.log({"epochs": epoch, "validation/f1": f1.mean()})


# if __name__ == "__main__":
#     run()
