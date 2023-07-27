"""
Script for training a single model for OOD detection.
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import torch
from torch import optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

# Import the network models
from resnet_cNfc import resnet18

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

    parser.add_argument("--log-interval", type=int, default=log_interval, dest="log_interval", help="Log Interval on Terminal",)
    parser.add_argument("--save-interval", type=int, default=save_interval, dest="save_interval", help="Save Interval on Terminal",)
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name, dest="saved_model_name", help="file name of the pre-trained model",)
    parser.add_argument("--save-path", type=str, default=save_loc, dest="save_loc", help="Path to export the model",)

    parser.add_argument("--first-milestone", type=int, default=first_milestone, dest="first_milestone", help="First milestone to change lr",)
    parser.add_argument("--second-milestone", type=int, default=second_milestone, dest="second_milestone", help="Second milestone to change lr",)

    return parser

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

grid_size = (91, 91) # 91: 0/45/90 0/30/60/90 or 85: 0/42/84 0/28/56/84    45m / 2.5s = 18m/s = 64.8km/h = 40.2648mph    40mph = 64.3737km/h    35mph = 56.3270km/h = 15.6463m/s * 2.5s = 39.1159

# ========================================================================================== #

if __name__ == "__main__":
    # Parsing the arguments
    args = training_args().parse_args()

    # Setting the seed
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    # Setting the device
    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    # Setting the num_outputs from dataset
    # num_outputs = dataset_num_outputs[args.dataset]
    num_outputs = grid_size

    # Choosing the model to train
    net = models[args.model](
        num_outputs = num_outputs,)

    # Using the gpu
    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Choosing the optimizer
    opt_params = net.parameters()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(opt_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1)

    # Drawing the map image for training
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

    map = np.full((4096, 4096, 3), 128, np.uint8)
    compensator = np.array([200, 256])
    for lane in lanes:
        for i, _ in enumerate(lane[:-1]):
            dx = lane[i+1][0] - lane[i][0]
            dy = lane[i+1][1] - lane[i][1]
            r = np.sqrt(dx * dx + dy * dy)
            if r > 0.1:
                color = ( int(dx * 127 / r + 128), 128, int(dy * 127 / r + 128) )
                cv2.line(map, ((lane[i] + compensator) * 8.).astype(np.int32), ((lane[i+1] + compensator) * 8.).astype(np.int32), color, 4)

    # ============================== #
    
    # Creating the summary writer in tensorboard
    writer = SummaryWriter(args.save_loc + "stats_logging/")
    training_set_loss = {}
    save_name = str(args.model) + str(args.seed)
    print("Model save name", save_name)
    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        
        # Loading the matrix dataset for preprocessing
        record = np.load("data/log_speed30/" + str(random.randrange(1000)) + ".npy") # (5000, number_of_vehicles, [location.x, locataion.y, rotation.yaw, v.x, v.y]))
        record_index = list(range(1, np.shape(record)[0] - 50))
        random.shuffle(record_index)
        
        # Sampling the 100 indexes from 0 to 4950
        num_samples = 100
        training_epoch_loss = 0
        for step in record_index[:num_samples]:
            map_copied = map.copy()
            current_record = record[step] # (number_of_vehicles, [location.x, locataion.y, rotation.yaw, v.x, v.y]), x,y: meter, yaw: -180~180deg, v: m/s
            current_xy = current_record[:, :2]
            current_yaw = np.reshape(current_record[:, 2], (265, 1)) # positive yaw: counterclockwise, negative yaw: clockwise
            after_10_xy = record[step+10, :, :2]        
            after_30_xy = record[step+30, :, :2]
            after_50_xy = record[step+50, :, :2]
            combined_record = np.concatenate((current_xy, current_yaw, after_10_xy, after_30_xy, after_50_xy), axis=1)

            # Generating the grid labels by preprocessing
            grid_label_after_10_array = []
            grid_label_after_30_array = []
            grid_label_after_50_array = []
            for cr in combined_record:                
                current_x, current_y, current_yaw, after_10_x, after_10_y, after_30_x, after_30_y, after_50_x, after_50_y = cr

                # Rotating the heading of vehicle to align with center-top cell of grid
                dx_10 = after_10_x - x
                dy_10 = after_10_y - y
                dx_30 = after_30_x - x
                dy_30 = after_30_y - y
                dx_50 = after_50_x - x
                dy_50 = after_50_y - y
                yaw_radian = np.radians(yaw)                
                after_10_x_rotated = -dx_10 * np.sin(yaw_radian) + dy_10 * np.cos(yaw_radian)
                after_10_y_rotated = dx_10 * np.cos(yaw_radian) + dy_10 * np.sin(yaw_radian)                
                after_30_x_rotated = -dx_30 * np.sin(yaw_radian) + dy_30 * np.cos(yaw_radian)
                after_30_y_rotated = dx_30 * np.cos(yaw_radian) + dy_30 * np.sin(yaw_radian)                
                after_50_x_rotated = -dx_50 * np.sin(yaw_radian) + dy_50 * np.cos(yaw_radian)
                after_50_y_rotated = dx_50 * np.cos(yaw_radian) + dy_50 * np.sin(yaw_radian)

                grid_after_10_x = int(grid_size[0] // 2 + round(after_10_x_rotated))
                grid_after_10_y = int(grid_size[1] // 2 + round(after_10_y_rotated))
                grid_after_30_x = int(grid_size[0] // 2 + round(after_30_x_rotated))
                grid_after_30_y = int(grid_size[1] // 2 + round(after_30_y_rotated))
                grid_after_50_x = int(grid_size[0] // 2 + round(after_50_x_rotated))
                grid_after_50_y = int(grid_size[1] // 2 + round(after_50_y_rotated))

                """
                print(f"After 10: ({grid_after_10_x}, {grid_after_10_y})    After 30: ({grid_after_30_x}, {grid_after_30_y})    After 50: ({grid_after_50_x}, {grid_after_50_y})")
                """
                if not (0 <= grid_after_10_x < grid_size[0] and 0 <= grid_after_10_y < grid_size[1]):
                    raise ValueError(f"Location after 10 timestep: ({grid_after_10_x}, {grid_after_10_y}) is outside the grid")
                if not (0 <= grid_after_30_x < grid_size[0] and 0 <= grid_after_30_y < grid_size[1]):
                    raise ValueError(f"Location after 30 timestep: ({grid_after_30_x}, {grid_after_30_y}) is outside the grid")
                if not (0 <= grid_after_50_x < grid_size[0] and 0 <= grid_after_50_y < grid_size[1]):
                    raise ValueError(f"Location after 50 timestep: ({grid_after_50_x}, {grid_after_50_y}) is outside the grid")
                
                # Saving the grid label by stacking as array
                grid_label_after_10 = np.zeros(grid_size)
                grid_label_after_30 = np.zeros(grid_size)
                grid_label_after_50 = np.zeros(grid_size)
                grid_label_after_10[grid_after_10_x, grid_after_10_y] = 1
                grid_label_after_30[grid_after_30_x, grid_after_30_y] = 1
                grid_label_after_50[grid_after_50_x, grid_after_50_y] = 1
                grid_label_after_10_array.append(grid_label_after_10) # (number_of_vehicles, grid_size[0], grid_size[1])                
                grid_label_after_30_array.append(grid_label_after_30) # (number_of_vehicles, grid_size[0], grid_size[1])                
                grid_label_after_50_array.append(grid_label_after_50) # (number_of_vehicles, grid_size[0], grid_size[1])
                
                # Visualizing the grid label
                """
                if grid_after_10_x == grid_after_30_x == grid_after_50_x and grid_after_10_y == grid_after_30_y == grid_after_50_y:
                    pass
                else:                
                    checkerboard_background = np.indices(grid_size).sum(axis=0) % 2
                    custom_color_map = mcolors.LinearSegmentedColormap.from_list("Custom", [(0, "silver"), (1, "white")], N=2)
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(checkerboard_background, cmap=custom_color_map, origin='lower')
                    ax.plot(grid_size[0] // 2, grid_size[1] // 2, 'ro')
                    ax.plot(grid_after_10_x, grid_after_10_y, 'yo')
                    ax.plot(grid_after_30_x, grid_after_30_y, 'go')
                    ax.plot(grid_after_50_x, grid_after_50_y, 'bo')                    
                    plt.show()
            print("grid_label_after_10_array shape :", np.array(grid_label_after_10_array).shape) # (number_of_vehicles, grid_size[0], grid_size[1])
            """
            
            # Generating the map inputs by preprocessing
            for cr in current_record:
                cv2.circle(map_copied, tuple(((cr[:2] + compensator) * 8.).astype(int)), 12, (128, 255, 128), -1)
            map_input_array = []
            map_cropping_size = 300
            for cr in current_record:
                position = (cr[:2] + compensator) * 8.
                M1 = np.float32( [ [1, 0, -position[0]], [0, 1, -position[1]], [0, 0, 1] ] )
                M2 = cv2.getRotationMatrix2D((0, 0), s[2] + 90, 1.0)
                M2 = np.append(M2, np.float32([[0, 0, 1]]), axis=0)
                M3 = np.float32( [ [1, 0, map_cropping_size/2], [0, 1, map_cropping_size*3/4], [0, 0, 1] ] )
                M = np.matmul(np.matmul(M3, M2), M1)
                map_rotated_n_cropped = cv2.warpAffine(map_copied, M[:2], (map_cropping_size, map_cropping_size))
                map_input_array.append(map_rotated_n_cropped.astype(np.float32) / 128.0 - 1.0) # (number_of_vehicles, map_cropping_size, map_cropping_size, 3)
                
                # Visualizing the map
                """
                map_rotated_n_cropped = cv2.cvtColor(map_rotated_n_cropped, cv2.COLOR_BGR2RGB)
                plt.imshow(map_rotated_n_cropped)
                plt.axis('off')
                plt.show()
            print("map_input_array shape :", np.array(map_input_array).shape) # (number_of_vehicles, map_cropping_size, map_cropping_size, 3)
            """

            map_input_tensor = torch.tensor(map_input_array).to(device) # (number_of_vehicles, map_cropping_size, map_cropping_size, 3)
            record_input_tensor = torch.tensor(current_record).to(device) # (number_of_vehicles, [location.x, locataion.y, rotation.yaw, v.x, v.y])
            grid_label_after_10_tensor = torch.tensor(grid_label_after_10_array).to(device) # (number_of_vehicles, grid_size[0], grid_size[1])
            grid_label_after_30_tensor = torch.tensor(grid_label_after_30_array).to(device) # (number_of_vehicles, grid_size[0], grid_size[1])
            grid_label_after_50_tensor = torch.tensor(grid_label_after_50_array).to(device) # (number_of_vehicles, grid_size[0], grid_size[1])
            
            net.train()
            optimizer.zero_grad()
            output_after_10, output_after_30, output_after_50 = net(map_input_tensor, record_input_tensor)

            # Flattening the output and label
            output_after_10_flattened = output_after_10.view(output_after_10.size(0), -1)
            output_after_30_flattened = output_after_30.view(output_after_30.size(0), -1)
            output_after_50_flattened = output_after_50.view(output_after_50.size(0), -1)
            label_after_10_flattened = grid_label_after_10_tensor.view(grid_label_after_10_tensor.size(0), -1)
            label_after_30_flattened = grid_label_after_30_tensor.view(grid_label_after_30_tensor.size(0), -1)
            label_after_50_flattened = grid_label_after_50_tensor.view(grid_label_after_50_tensor.size(0), -1)
            
            # Calculating the cross entropy loss by applying the softmax output 
            loss_function_dict = {"cross_entropy": F.cross_entropy}
            cross_entropy_loss_1 = loss_function_dict[args.loss_function](output_after_10_flattened, label_after_10_flattened) # 0 ~ inf
            cross_entropy_loss_2 = loss_function_dict[args.loss_function](output_after_30_flattened, label_after_30_flattened) # 0 ~ inf
            cross_entropy_loss_3 = loss_function_dict[args.loss_function](output_after_50_flattened, label_after_50_flattened) # 0 ~ inf

            # Calculating the euclidean distance loss
            _, output_after_10_indices = torch.max(output_after_10_flattened, dim=1)
            _, label_after_10_indices = torch.max(label_after_10_flattened, dim=1)
            _, output_after_30_indices = torch.max(output_after_30_flattened, dim=1)
            _, label_after_30_indices = torch.max(label_after_30_flattened, dim=1)
            _, output_after_50_indices = torch.max(output_after_50_flattened, dim=1)
            _, label_after_50_indices = torch.max(label_after_50_flattened, dim=1)
            
            output_after_10_cell = torch.stack((output_after_10_indices // 91, output_after_10_indices % 91), dim=1)
            label_after_10_cell = torch.stack((label_after_10_indices // 91, label_after_10_indices % 91), dim=1)
            output_after_30_cell = torch.stack((output_after_30_indices // 91, output_after_30_indices % 91), dim=1)
            label_after_30_cell = torch.stack((label_after_30_indices // 91, label_after_30_indices % 91), dim=1)
            output_after_50_cell = torch.stack((output_after_50_indices // 91, output_after_50_indices % 91), dim=1)
            label_after_50_cell = torch.stack((label_after_50_indices // 91, label_after_50_indices % 91), dim=1)
            
            euclidean_distance_loss_1 = torch.norm(output_after_10_cell.float() - label_after_10_cell.float(), dim=1).mean() # 0 ~ 128.062 (sqrt(90^2 + 90^2))
            euclidean_distance_loss_2 = torch.norm(output_after_30_cell.float() - label_after_30_cell.float(), dim=1).mean() # 0 ~ 128.062 (sqrt(90^2 + 90^2))
            euclidean_distance_loss_3 = torch.norm(output_after_50_cell.float() - label_after_50_cell.float(), dim=1).mean() # 0 ~ 128.062 (sqrt(90^2 + 90^2))
            
            training_step_loss = 3/10*cross_entropy_loss_1 + 3/10*cross_entropy_loss_2 + 3/10*cross_entropy_loss_3 + 1/30*euclidean_distance_loss_1 + 1/30*euclidean_distance_loss_2 + 1/30*euclidean_distance_loss_3
            
            training_step_loss.backward()
            training_epoch_loss += training_step_loss.item()
            # Updating the parameters
            optimizer.step()

        training_epoch_loss /= num_samples # / 100
        print("====> Epoch: {} loss: {:.4f}".format(epoch, training_epoch_loss))
        writer.add_scalar(save_name + "_training_epoch_loss", training_epoch_loss, (epoch + 1))
        training_set_loss[epoch] = training_epoch_loss

        # Decaying the learning_rate according to milestones
        scheduler.step()
        
        # Saving the model per save_interval
        if (epoch + 1) % args.save_interval == 0:
            saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
            torch.save(net.state_dict(), saved_name)

    # Saving the model before completion
    saved_name = args.save_loc + save_name + "_" + str(epoch + 1) + ".model"
    torch.save(net.state_dict(), saved_name)
    print("Model saved to ", saved_name)
    
    with open(saved_name[: saved_name.rfind("_")] + "_training_set_loss.json", "a") as f:
        json.dump(training_set_loss, f)
    
    writer.close()
