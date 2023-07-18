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

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter

# ========================================================================================== #

def training_args():
    default_dataset = "cifar10"
    dataset_root = "./"
    ood_dataset = "svhn"
    train_batch_size = 128
    test_batch_size = 128

    learning_rate = 0.1
    momentum = 0.9
    optimizer = "sgd"
    loss = "cross_entropy"
    weight_decay = 5e-4
    log_interval = 50
    save_interval = 25
    save_loc = "./"
    saved_model_name = "resnet18_350.model"
    epoch = 350
    first_milestone = 150  # Milestone for change in lr
    second_milestone = 250  # Milestone for change in lr

    model = "resnet18"

    parser = argparse.ArgumentParser(description="Args for training parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument("--dataset", type=str, default=default_dataset, dest="dataset", help="dataset to train on",)
    parser.add_argument("--dataset-root", type=str, default=dataset_root, dest="dataset_root", help="path of a dataset (useful for dirty mnist)",)

    parser.add_argument("-b", type=int, default=train_batch_size, dest="train_batch_size", help="Batch size",)

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")

    parser.add_argument("-e", type=int, default=epoch, dest="epoch", help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=learning_rate, dest="learning_rate", help="Learning rate",)
    parser.add_argument("--mom", type=float, default=momentum, dest="momentum", help="Momentum")
    parser.add_argument("--nesterov", action="store_true", dest="nesterov", help="Whether to use nesterov momentum in SGD",)
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay, dest="weight_decay", help="Weight Decay",)
    parser.add_argument("--opt", type=str, default=optimizer, dest="optimizer", help="Choice of optimisation algorithm",)

    parser.add_argument("--loss", type=str, default=loss, dest="loss_function", help="Loss function to be used for training",)
    parser.add_argument("--loss-mean", action="store_true", dest="loss_mean", help="whether to take mean of loss instead of sum to train",)
    parser.set_defaults(loss_mean=False)

    parser.add_argument("--log-interval", type=int, default=log_interval, dest="log_interval", help="Log Interval on Terminal",)
    parser.add_argument("--save-interval", type=int, default=save_interval, dest="save_interval", help="Save Interval on Terminal",)
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name, dest="saved_model_name", help="file name of the pre-trained model",)
    parser.add_argument("--save-path", type=str, default=save_loc, dest="save_loc", help="Path to export the model",)

    parser.add_argument("--first-milestone", type=int, default=first_milestone, dest="first_milestone", help="First milestone to change lr",)
    parser.add_argument("--second-milestone", type=int, default=second_milestone, dest="second_milestone", help="Second milestone to change lr",)

    return parser

# ============================================================ #

def get_train_valid_loader(batch_size, val_seed, val_size=0.1, num_workers=4, pin_memory=False, **kwargs):
    """
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme mentioned in the paper. Only applied on the train split.
    - val_seed: fix seed for reproducibility.
    - val_size: percentage split of the training set used for the validation set. Should be a float in the range [0, 1].
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    """
    error_msg = "[!] val_size should be in the range [0, 1]."
    assert (val_size >= 0) and (val_size <= 1), error_msg

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)

    # define transforms
    train_transform = transforms.Compose([transforms.ToTensor(), normalize,])
    valid_transform = transforms.Compose([transforms.ToTensor(), normalize,])

    # load the dataset
    data_dir = "./data"
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform,)
    valid_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=valid_transform,)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.seed(val_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(valid_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True,)
    valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False,)

    return (train_loader, valid_loader)

# ============================================================ #

def train_single_epoch(epoch, model, train_loader, optimizer, device, loss_function="cross_entropy", loss_mean=False,):
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = loss_function_dict[loss_function](logits, labels)

        if loss_mean:
            loss = loss / len(data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader) * len(data), 100.0 * batch_idx / len(train_loader), loss.item(),))

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / num_samples))
    return train_loss / num_samples

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

# ========================================================================================== #

if __name__ == "__main__":
    # Parsing arguments
    args = training_args().parse_args()

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
        val_size=0.1,
        val_seed=args.seed,
        pin_memory=args.gpu,)

    # Creating summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")
    training_set_loss = {}
    save_name = str(args.model) + str(args.seed)
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
