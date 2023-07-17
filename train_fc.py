"""
Script for training a single model for OOD detection.
"""

import json
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.dirty_mnist as dirty_mnist

# Import network models
from net.resnet import resnet18, resnet50

# Import train and validation utilities
from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.train_utils import model_save_name
from utils.train_utils import train_single_epoch, test_single_epoch

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter

# ========================================================================================== #

dataset_num_outputs = {"cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "dirty_mnist": dirty_mnist,}

models = {
    "resnet18": resnet18,
    "resnet50": resnet50,}

# ============================================================ #

if __name__ == "__main__":
    # Parsing arguments
    args = training_args().parse_args() # utils.args training_args()

    # Setting seed
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    # Setting device
    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    # Setting num_outputs from dataset
    num_outputs = dataset_num_outputs[args.dataset]

    # Choosing model to train
    net = models[args.model](
        num_outputs = num_outputs,)

    # Using gpu
    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Choosing optimizer
    opt_params = net.parameters()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(opt_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1)

    # Loading train dataset
    train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
        root=args.dataset_root,
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        val_size=0.1,
        val_seed=args.seed,
        pin_memory=args.gpu,)

    # Creating summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")
    training_set_loss = {}
    save_name = model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)
    print("Model save name", save_name)

    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        train_loss = train_single_epoch(epoch, net, train_loader, optimizer, device, loss_function=args.loss_function, loss_mean=args.loss_mean,)
        training_set_loss[epoch] = train_loss
        writer.add_scalar(save_name + "_train_loss", train_loss, (epoch + 1))

        # Decaying learning_rate in every epoch.
        scheduler.step()

        # Saving model
        if (epoch + 1) % args.save_interval == 0:
            saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
            torch.save(net.state_dict(), saved_name)

    # Saving model
    saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
    torch.save(net.state_dict(), saved_name)
    print("Model saved to ", saved_name)
    
    with open(saved_name[: saved_name.rfind("_")] + "_train_loss.json", "a") as f:
        json.dump(training_set_loss, f)
    
    writer.close()
