"""
Script for training a single model for OOD detection.
"""

import json
import numpy as np
import cv2
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

def train_single_epoch(epoch, model, train_loader, optimizer, device, loss_function="cross_entropy", loss_mean=False,):
    log_interval = 10
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader): # 데이터셋을 가지고
        data = data.to(device)
        labels = labels.to(device) # label

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

    # Loading image dataset for training
    lanes = []
    with open("lane_town03.txt", "rt") as rf:
        for line in rf.readlines():
            lane = []
            for s in line.split("\t"):
                v = s.split(",")
                if len(v) == 2:
                    lane.append([float(v[0]), float(v[1])])
            if len(lane) > 0:
                lanes.append(np.array(lane))
    
    map = np.full((4096, 4096, 3), 256, np.uint8)
    compensator = np.array([200, 256])
    for lane in lanes:
        for i, _ in enumerate(lane[:-1]):
            dx = lane[i+1][0] - lane[i][0]
            dy = lane[i+1][1] - lane[i][1]
            r = np.sqrt(dx * dx + dy * dy)
            if r > 0.1:
                color = ( int(dx * 127 / r + 128), 128, int(dy * 127 / r + 128) )
                cv2.line(map, ((lane[i] + compensator) * 8.).astype(np.int32), ((lane[i+1] + compensator) * 8.).astype(np.int32), color, 4)
    
    # Creating summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")
    training_set_loss = {}
    save_name = str(args.model) + str(args.seed)
    print("Model save name", save_name)

    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        
        # Loading matrix dataset for training
        record = np.load("gathered/log1/" + str(random.randrange(1000)) + ".npy")
        record_index = list(range(1, np.shape(record)[0] - 50))
        random.shuffle(record_index)
        for step in record_index[:100]:
            map_copied = map.copy()
            current_record = record[step] # [location.x, locataion.y, rotation.yaw, v.x, v.y] x number of vehicles
            
            for cr in current_record:
                cv2.circle(map_copied, tuple(((cr[:2] + compensator) * 8.).astype(int)), 12, (128, 255, 128), -1)

            map_array = []
            map_cropping_size = 300
            for cr in cur_record:
                position = (cr[:2] + compensator) * 8.
                M1 = np.float32( [ [1, 0, -position[0]], [0, 1, -position[1]], [0, 0, 1] ] )
                M2 = cv2.getRotationMatrix2D((0, 0), s[2] + 90, 1.0)
                M2 = np.append(M2, np.float32([[0, 0, 1]]), axis=0)
                M3 = np.float32( [ [1, 0, map_cropping_size/2], [0, 1, map_cropping_size*3/4], [0, 0, 1] ] )
                M = np.matmul(np.matmul(M3, M2), M1)
                map_rotated = cv2.warpAffine(map_copied, M[:2], (map_cropping_size, map_cropping_size))
                map_array.append(map_rotated.astype(np.float32) / 255.)
        
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
