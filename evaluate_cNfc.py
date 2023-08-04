"""
Script to evaluate a single model. 
"""

import os
import json
import numpy as np
import random
import math
import torch
from torch import nn
from torch.nn import functional as F
import argparse
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from resnet_cNfc import resnet18

from sklearn.metrics import accuracy_score
from utils.temperature_scaling_cNfc import ModelWithTemperature

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ========================================================================================== #

def evaluation_args():
    model = "resnet18"
    sn_coeff = 3.0
    runs = 1
    model_type = "softmax"

    parser = argparse.ArgumentParser(description="Training for calibration.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=False)

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")
    parser.add_argument("--runs", type=int, default=runs, dest="runs", help="Number of models to aggregate over",)

    parser.add_argument("--model-type", type=str, default=model_type, choices=["softmax", "ensemble", "gmm"], dest="model_type", help="Type of model to load for evaluation.",)

    return parser

# ========================================================================================== #

# For GMM and softmax
def test_classification_net(model, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset.
    """
    # Getting the logits and labels
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            logit = model(data)
            logits.append(logit)
            labels.append(label)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    # ============================== #
    
    # Getting the softmax vectors given logits from model
    softmax_prob = F.softmax(logits, dim=1)

    # ============================== #
    
    # Getting the classification accuracy and confusion matrix given softmax vectors and labels from model
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    confidence_vals, predictions = torch.max(softmax_prob, dim=1)
    labels_list.extend(labels.cpu().numpy())
    predictions_list.extend(predictions.cpu().numpy())
    confidence_vals_list.extend(confidence_vals.cpu().numpy())
    accuracy = accuracy_score(labels_list, predictions_list)
    return (accuracy,
            labels_list,
            predictions_list,
            confidence_vals_list,)
    
# ========================================================================================== #

# For GMM, softmax and ensemble
def expected_calibration_error(confs, preds, labels, num_bins=10):
    # Initializing the bins
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
        bin_dict[i]["count"] = 0
        bin_dict[i]["conf"] = 0
        bin_dict[i]["acc"] = 0
        bin_dict[i]["bin_acc"] = 0
        bin_dict[i]["bin_conf"] = 0

    # ============================== #
    
    # Populating the bins
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn]["count"] = bin_dict[binn]["count"] + 1
        bin_dict[binn]["conf"] = bin_dict[binn]["conf"] + confidence
        bin_dict[binn]["acc"] = bin_dict[binn]["acc"] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if bin_dict[binn]["count"] == 0:
            bin_dict[binn]["bin_acc"] = 0
            bin_dict[binn]["bin_conf"] = 0
        else:
            bin_dict[binn]["bin_acc"] = float(bin_dict[binn]["acc"]) / bin_dict[binn]["count"]
            bin_dict[binn]["bin_conf"] = bin_dict[binn]["conf"] / float(bin_dict[binn]["count"])

    # ============================== #
    
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i]["bin_acc"]
        bin_confidence = bin_dict[i]["bin_conf"]
        bin_count = bin_dict[i]["count"]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)
    return ece

# ========================================================================================== #

# For GMM
def get_embeddings(net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels

# ========================================================================================== #

JITTERS = [0, torch.finfo(torch.double).tiny] + [10 ** exp for exp in range(-308, 0, 1)]
def _centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res

def gmm_fit(embeddings, labels, num_classes):
    with torch.no_grad():
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        classwise_cov_features = torch.stack([_centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)])

    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(classwise_cov_features.shape[1], device=classwise_cov_features.device,).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),)
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
            break

    return gmm, jitter_eps

# ========================================================================================== #

def _gmm_forward(net, gaussians_model, data_B_X):
    if isinstance(net, nn.DataParallel):
        features_B_Z = net.module(data_B_X)
        features_B_Z = net.module.feature
    else:
        features_B_Z = net(data_B_X)
        features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])

    return log_probs_B_Y

def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = _gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N
    
# ========================================================================================== #
  
# For GMM and softmax
def entropy(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy

# ========================================================================================== #

# For GMM and softmax
# logsumexp() is sensitive to greater value because of exp(). It can show similar effect to using mean() after max().
def logsumexp(logits):
    return torch.logsumexp(logits, dim=1, keepdim=False)

# ========================================================================================== #

# For softmax
def get_roc_auc(net, test_loader, ood_test_loader, uncertainty, device, confidence=False):
    logits, _ = get_logits_labels(net, test_loader, device)
    ood_logits, _ = get_logits_labels(net, ood_test_loader, device)

    return get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=confidence)

# ========================================================================================== #

# For GMM and softmax
def get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=False):
    # For entropy and logsumexp
    uncertainties = uncertainty(logits)
    ood_uncertainties = uncertainty(ood_logits)
    
    # For in-distribution
    bin_labels = torch.zeros(uncertainties.shape[0]).to(device) # binary labels
    in_scores = uncertainties
    
    # For out-of-distribution
    bin_labels = torch.cat((bin_labels, torch.ones(ood_uncertainties.shape[0]).to(device)))
    ood_scores = ood_uncertainties
    
    # Preprocessing for logsumexp, not entropy
    if confidence:
        bin_labels = 1 - bin_labels
    
    scores = torch.cat((in_scores, ood_scores))
    
    # Calculating the fpr(false positive rate), tpr(true positive rate), and thresholds
    fpr, tpr, thresholds = metrics.roc_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels.cpu().numpy(), scores.cpu().numpy())

    return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc

# ========================================================================================== #

models = {
    "resnet18": resnet18,}

grid_size = (127, 127) # 97: 0/48/96 0/32/64/96 or 91: 0/45/90 0/30/60/90 or 85: 0/42/84 0/28/56/84    45m / 2.5s = 18m/s = 64.8km/h = 40.2648mph    40mph = 64.3737km/h    35mph = 56.3270km/h = 15.6463m/s * 2.5s = 39.1159

model_to_num_dim = {"resnet50": 2048, "wide_resnet": 640, "vgg16": 512}

# ========================================================================================== #

if __name__ == "__main__":
    # Parsing the arguments
    args = evaluation_args().parse_args()

    # Setting the seed
    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    # Setting the device
    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    # Setting the num_outputs from dataset
    num_outputs = grid_size

    # test_loader = dataset_loader[args.dataset].get_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)
    # ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)

    # Evaluating the models
    accuracies = []
    
    # m1 - Uncertainty/Confidence Metric 1 for deterministic model: logsumexp
    #                                      for ensemble: entropy
    # m2 - Uncertainty/Confidence Metric 2 for deterministic model: entropy
    #                                      for ensemble: MI
    # For model
    eces = []
    m1_aurocs = []
    m1_auprcs = []
    m2_aurocs = []
    m2_auprcs = []

    # For temperature scaled model
    t_eces = []
    t_m1_aurocs = []
    t_m1_auprcs = []
    t_m2_aurocs = []
    t_m2_auprcs = []

    # ============================== #
    
    # Drawing the map image for training
    lanes = []
    with open("data/lane_town03.txt", "rt") as rf:
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
    
    for i in range(args.runs):
        print(f"========== Run {i+1} ==========")
        
        # Choosing the model to evaluate
        # train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(batch_size=args.batch_size, augment=args.data_aug, val_seed=(args.seed+i), val_size=0.1, pin_memory=args.gpu,)
        num_outputs = grid_size
        net = models[args.model](
            num_outputs=num_outputs)
            # temp=1.0,)

        # ============================== #
        
        # Using the gpu
        if args.gpu:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        # ============================== #

        # Loading the saved model to evaluate
        net.load_state_dict(torch.load("./save/resnet183_130.model"))
        net.eval()

        # ============================== #
        
        # Loading the matrix dataset for preprocessing
        record = np.load("data/log_speed30/" + str(random.randrange(1000)) + ".npy") # (5000, number of vehicles spawned, [location.x, locataion.y, rotation.yaw, v.x, v.y]))
        record_index_shuffled = list(range(1, np.shape(record)[0] - 50))
        random.shuffle(record_index_shuffled)

        # ============================== #

        # Sampling the 100 indices from 0 to 4950
        num_index_samples = 1
        num_vehicle_samples = 50 # Vehicles are spawned in random points for each iteration.
        for step in record_index_shuffled[:num_index_samples]:
            current_record = record[step]
            current_record_sampled = record[step][:num_vehicle_samples] # (num_vehicle_samples, [location.x, locataion.y, rotation.yaw, v.x, v.y]), x,y: meter    yaw: -180~180deg    v: m/s

            current_xy = current_record_sampled[:, :2]
            current_yaw = np.reshape(current_record_sampled[:, 2], (num_vehicle_samples, 1)) # positive yaw: counterclockwise, negative yaw: clockwise
            current_velocity_xy = current_record_sampled[:, 3:]
            after_10_xy = record[step+10, :num_vehicle_samples, :2]        
            after_30_xy = record[step+30, :num_vehicle_samples, :2]
            after_50_xy = record[step+50, :num_vehicle_samples, :2]
            combined_record_sampled = np.concatenate((current_xy, current_yaw, current_velocity_xy, after_10_xy, after_30_xy, after_50_xy), axis=1)

            # Generating the grid labels by preprocessing
            grid_label_after_10_array = []
            grid_label_after_30_array = []
            grid_label_after_50_array = []
            counter_exclude = 0
            counter_include = 0
            counter_exclude_array = []
            for cr in combined_record_sampled:
                current_x, current_y, current_yaw, current_velocity_x, current_velocity_y, after_10_x, after_10_y, after_30_x, after_30_y, after_50_x, after_50_y = cr

                velocity = math.sqrt(current_velocity_x**2 + current_velocity_y**2) * 3.6
                
                # Rotating the heading of vehicle to align with center-top cell of grid
                dx_10 = after_10_x - current_x
                dy_10 = after_10_y - current_y
                dx_30 = after_30_x - current_x
                dy_30 = after_30_y - current_y
                dx_50 = after_50_x - current_x
                dy_50 = after_50_y - current_y
                current_yaw_radian = np.radians(current_yaw)
                after_10_x_rotated = -dx_10 * np.sin(current_yaw_radian) + dy_10 * np.cos(current_yaw_radian)
                after_10_y_rotated = dx_10 * np.cos(current_yaw_radian) + dy_10 * np.sin(current_yaw_radian)                
                after_30_x_rotated = -dx_30 * np.sin(current_yaw_radian) + dy_30 * np.cos(current_yaw_radian)
                after_30_y_rotated = dx_30 * np.cos(current_yaw_radian) + dy_30 * np.sin(current_yaw_radian)                
                after_50_x_rotated = -dx_50 * np.sin(current_yaw_radian) + dy_50 * np.cos(current_yaw_radian)
                after_50_y_rotated = dx_50 * np.cos(current_yaw_radian) + dy_50 * np.sin(current_yaw_radian)

                grid_after_10_x = int(grid_size[0] // 2 + round(after_10_x_rotated))
                grid_after_10_y = int(grid_size[1] // 2 + round(after_10_y_rotated))
                grid_after_30_x = int(grid_size[0] // 2 + round(after_30_x_rotated))
                grid_after_30_y = int(grid_size[1] // 2 + round(after_30_y_rotated))
                grid_after_50_x = int(grid_size[0] // 2 + round(after_50_x_rotated))
                grid_after_50_y = int(grid_size[1] // 2 + round(after_50_y_rotated))

                # print(f"After 10: ({grid_after_10_x}, {grid_after_10_y})    After 30: ({grid_after_30_x}, {grid_after_30_y})    After 50: ({grid_after_50_x}, {grid_after_50_y})")
                
                # ============================== #
                
                # Filtering out some data of stationary vehicles
                if grid_after_10_x == grid_after_30_x == grid_after_50_x and grid_after_10_y == grid_after_30_y == grid_after_50_y:
                    if counter_include % 4 == 0:
                        counter_include += 1
                    else:
                        counter_exclude_array.append(counter_exclude)
                        counter_exclude += 1
                        counter_include += 1
                        continue
                        
                # ============================== #
                
                # Filtering the label outside the grid
                if not (0 <= grid_after_10_x < grid_size[0] and 0 <= grid_after_10_y < grid_size[1]):
                    counter_exclude_array.append(counter_exclude)
                    counter_exclude += 1
                    print(f"Raw location current: ({current_x:.4f}, {current_y:.4f})")
                    print(f"Raw location after 10 timestep: ({after_10_x:.4f}, {after_10_y:.4f})")
                    print(f"Raw location after 30 timestep: ({after_30_x:.4f}, {after_30_y:.4f})")
                    print(f"Raw location after 50 timestep: ({after_50_x:.4f}, {after_50_y:.4f})")
                    print(f"Yaw current: {current_yaw:.4f}")
                    print(f"Grid Location after 10 timestep: ({grid_after_10_x}, {grid_after_10_y}) is outside the grid. Current velocity is {velocity:.2f}km/h")                    
                    print(f"Grid Location after 30 timestep: ({grid_after_30_x}, {grid_after_30_y})")
                    print(f"Grid Location after 50 timestep: ({grid_after_50_x}, {grid_after_50_y})")
                    continue
                if not (0 <= grid_after_30_x < grid_size[0] and 0 <= grid_after_30_y < grid_size[1]):
                    counter_exclude_array.append(counter_exclude)
                    counter_exclude += 1
                    print(f"Raw location current: ({current_x:.4f}, {current_y:.4f})")
                    print(f"Raw location after 10 timestep: ({after_10_x:.4f}, {after_10_y:.4f})")
                    print(f"Raw location after 30 timestep: ({after_30_x:.4f}, {after_30_y:.4f})")
                    print(f"Raw location after 50 timestep: ({after_50_x:.4f}, {after_50_y:.4f})")
                    print(f"Yaw current: {current_yaw:.4f}")
                    print(f"Grid Location after 10 timestep: ({grid_after_10_x}, {grid_after_10_y})")
                    print(f"Grid Location after 30 timestep: ({grid_after_30_x}, {grid_after_30_y}) is outside the grid. Current velocity is {velocity:.2f}km/h")
                    print(f"Grid Location after 50 timestep: ({grid_after_50_x}, {grid_after_50_y})")
                    continue
                if not (0 <= grid_after_50_x < grid_size[0] and 0 <= grid_after_50_y < grid_size[1]):
                    counter_exclude_array.append(counter_exclude)
                    counter_exclude += 1
                    print(f"Raw location current: ({current_x:.4f}, {current_y:.4f})")
                    print(f"Raw location after 10 timestep: ({after_10_x:.4f}, {after_10_y:.4f})")
                    print(f"Raw location after 30 timestep: ({after_30_x:.4f}, {after_30_y:.4f})")
                    print(f"Raw location after 50 timestep: ({after_50_x:.4f}, {after_50_y:.4f})")
                    print(f"Yaw current: {current_yaw:.4f}")
                    print(f"Grid Location after 10 timestep: ({grid_after_10_x}, {grid_after_10_y})")                    
                    print(f"Grid Location after 30 timestep: ({grid_after_30_x}, {grid_after_30_y})")
                    print(f"Grid Location after 50 timestep: ({grid_after_50_x}, {grid_after_50_y}) is outside the grid. Current velocity is {velocity:.2f}km/h")
                    continue
                    
                # ============================== #
                
                # Saving the grid label by stacking as array
                grid_label_after_10 = np.zeros(grid_size)
                grid_label_after_30 = np.zeros(grid_size)
                grid_label_after_50 = np.zeros(grid_size)
                grid_label_after_10[grid_after_10_x, grid_after_10_y] = 1
                grid_label_after_30[grid_after_30_x, grid_after_30_y] = 1
                grid_label_after_50[grid_after_50_x, grid_after_50_y] = 1
                grid_label_after_10_array.append(grid_label_after_10) # (num_vehicle_samples, grid_size[0], grid_size[1])
                grid_label_after_30_array.append(grid_label_after_30) # (num_vehicle_samples, grid_size[0], grid_size[1])
                grid_label_after_50_array.append(grid_label_after_50) # (num_vehicle_samples, grid_size[0], grid_size[1])
            
                # Visualizing the grid label
                checkerboard_background = np.indices(grid_size).sum(axis=0) % 2
                custom_color_map = mcolors.LinearSegmentedColormap.from_list("Custom", [(0, "silver"), (1, "white")], N=2)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(checkerboard_background, cmap=custom_color_map, origin='lower')
                ax.plot(grid_size[0] // 2, grid_size[1] // 2, 'ro')
                ax.plot(grid_after_10_x, grid_after_10_y, 'yo')
                ax.plot(grid_after_30_x, grid_after_30_y, 'go')
                ax.plot(grid_after_50_x, grid_after_50_y, 'bo')         
                plt.title(f"Label {counter_record}")
                plt.savefig(f"Label {counter_record}")
                # plt.show()
                
                # ============================== #
                
                # Increasing the number of counter
                counter_exclude += 1

            # ============================== #
            
            # Filtering the record data outside the grid
            current_record_sampled_filtered = np.delete(current_record_sampled, counter_record_filtering, axis=0)

            # Generating the map inputs by preprocessing
            map_copied = map.copy()
            # Drawing the circles representing location of vehicles on map for all vehicles, including unsampled ones
            for cr in current_record:
                cv2.circle(map_copied, tuple(((cr[:2] + compensator) * 8.).astype(int)), 12, (128, 255, 128), -1)
            map_input_array = []
            map_cropping_size = 300
            for cr in current_record_sampled_filtered:
                location = (cr[:2] + compensator) * 8.
                M1 = np.float32( [ [1, 0, -location[0]], [0, 1, -location[1]], [0, 0, 1] ] )
                M2 = cv2.getRotationMatrix2D((0, 0), cr[2] + 90, 1.0)
                M2 = np.append(M2, np.float32([[0, 0, 1]]), axis=0)
                M3 = np.float32( [ [1, 0, map_cropping_size/2], [0, 1, map_cropping_size*3/4], [0, 0, 1] ] )
                M = np.matmul(np.matmul(M3, M2), M1)
                map_rotated_n_cropped = cv2.warpAffine(map_copied, M[:2], (map_cropping_size, map_cropping_size)) # (width, height)
                map_input_array.append(map_rotated_n_cropped.astype(np.float32) / 128.0 - 1.0) # (num_vehicle_samples, map_cropping_size, map_cropping_size, 3)

            map_input_tensor = (torch.tensor(np.array(map_input_array), dtype=torch.float32).permute(0, 3, 1, 2)).to(device) # (num_vehicle_samples, map_cropping_size height, map_cropping_size width, 3 channels) → (num_vehicle_samples, 3 channels, map_cropping_size height, map_cropping_size width)
            record_input_tensor = torch.tensor(current_record_sampled_filtered, dtype=torch.float32).to(device) # (num_vehicle_samples, [location.x, locataion.y, rotation.yaw, v.x, v.y])
            grid_label_after_10_tensor = torch.tensor(np.array(grid_label_after_10_array)).to(device) # (num_vehicle_samples, grid_size[0], grid_size[1])
            grid_label_after_30_tensor = torch.tensor(np.array(grid_label_after_30_array)).to(device) # (num_vehicle_samples, grid_size[0], grid_size[1])
            grid_label_after_50_tensor = torch.tensor(np.array(grid_label_after_50_array)).to(device) # (num_vehicle_samples, grid_size[0], grid_size[1])
        
            output_after_10, output_after_30, output_after_50 = net(map_input_tensor, record_input_tensor)
            
            for i in range(num_vehicle_samples):
                output_after_10_max_coordinates = torch.argmax(output_after_10[i].view(-1))
                output_after_10_x = output_after_10_max_coordinates // 127
                output_after_10_y = output_after_10_max_coordinates % 127
                output_after_30_max_coordinates = torch.argmax(output_after_30[i].view(-1))
                output_after_30_x = output_after_30_max_coordinates // 127
                output_after_30_y = output_after_30_max_coordinates % 127
                output_after_50_max_coordinates = torch.argmax(output_after_50[i].view(-1))
                output_after_50_x = output_after_50_max_coordinates // 127
                output_after_50_y = output_after_50_max_coordinates % 127
                
                checkerboard_background = np.indices(grid_size).sum(axis=0) % 2
                custom_color_map = mcolors.LinearSegmentedColormap.from_list("Custom", [(0, "silver"), (1, "white")], N=2)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(checkerboard_background, cmap=custom_color_map, origin='lower')
                ax.plot(grid_size[0] // 2, grid_size[1] // 2, 'ro')
                ax.plot(output_after_10_x, output_after_10_y, 'yo')
                ax.plot(output_after_30_x, output_after_30_y, 'go')
                ax.plot(output_after_50_x, output_after_50_y, 'bo')
                plt.title(f"Output {i}")
                plt.savefig(f"Output {i}")
                #plt.show()
        
        # ============================== #

        # Evaluating the model
        # ece(expected calibration error) represents quality of aleatoric uncertainty. ece measures expectation of difference between accuracy and confidence
        (accuracy, labels_list, predictions, confidences,) = test_classification_net(net, test_loader, device)
        ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

        # Temperature scaling the trained model (Calibrating the classifier to adjust the confidence of prediction)
        temp_scaled_net = ModelWithTemperature(net)
        temp_scaled_net.set_temperature(val_loader)

        (t_accuracy, t_labels_list, t_predictions, t_confidences,) = test_classification_net(temp_scaled_net, test_loader, device)
        t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)

        # ============================== #
        
        if (args.model_type == "gmm"): # gmm
            # Evaluating the GMM model
            print("GMM Model")
            embeddings, labels = get_embeddings(net,
                                                train_loader,
                                                num_dim=model_to_num_dim[args.model],
                                                dtype=torch.double,
                                                device=device,
                                                storage_device=device,)

            try:
                gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
                logits, labels = gmm_evaluate(net, gaussians_model, test_loader, device=device, num_classes=num_classes, storage_device=device,)

                ood_logits, ood_labels = gmm_evaluate(net, gaussians_model, ood_test_loader, device=device, num_classes=num_classes, storage_device=device,)
                
                # auroc(area under receiver operating characteristic curve) represents quality of epistemic uncertainty.
                (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(logits, ood_logits, logsumexp, device, confidence=True)
                (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(logits, ood_logits, entropy, device)

                # In case of gmm, temperature scaling is meaningless.
                t_m1_auroc = m1_auroc
                t_m1_auprc = m1_auprc
                t_m2_auroc = m2_auroc
                t_m2_auprc = m2_auprc

            except RuntimeError as e:
                print("Runtime Error caught: " + str(e))
                continue

        # ============================== #
        
        else: # softmax
            # Evaluating the normal softmax model
            print("Softmax Model")
            (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc(net, test_loader, ood_test_loader, logsumexp, device, confidence=True)
            (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc(net, test_loader, ood_test_loader, entropy, device)

            (_, _, _), (_, _, _), t_m1_auroc, t_m1_auprc = get_roc_auc(temp_scaled_net, test_loader, ood_test_loader, logsumexp, device, confidence=True,)
            (_, _, _), (_, _, _), t_m2_auroc, t_m2_auprc = get_roc_auc(temp_scaled_net, test_loader, ood_test_loader, entropy, device)

        # ============================== #
        
        accuracies.append(accuracy)

        # Pre-temperature results
        eces.append(ece)
        m1_aurocs.append(m1_auroc)
        m1_auprcs.append(m1_auprc)
        m2_aurocs.append(m2_auroc)
        m2_auprcs.append(m2_auprc)

        # Post-temperature results
        t_eces.append(t_ece)
        t_m1_aurocs.append(t_m1_auroc)
        t_m1_auprcs.append(t_m1_auprc)
        t_m2_aurocs.append(t_m2_auroc)
        t_m2_auprcs.append(t_m2_auprc)

    # ============================== #
    
    accuracy_tensor = torch.tensor(accuracies)
    ece_tensor = torch.tensor(eces)
    m1_auroc_tensor = torch.tensor(m1_aurocs)
    m1_auprc_tensor = torch.tensor(m1_auprcs)
    m2_auroc_tensor = torch.tensor(m2_aurocs)
    m2_auprc_tensor = torch.tensor(m2_auprcs)

    t_ece_tensor = torch.tensor(t_eces)
    t_m1_auroc_tensor = torch.tensor(t_m1_aurocs)
    t_m1_auprc_tensor = torch.tensor(t_m1_auprcs)
    t_m2_auroc_tensor = torch.tensor(t_m2_aurocs)
    t_m2_auprc_tensor = torch.tensor(t_m2_auprcs)

    mean_accuracy = torch.mean(accuracy_tensor)
    mean_ece = torch.mean(ece_tensor)
    mean_m1_auroc = torch.mean(m1_auroc_tensor)
    mean_m1_auprc = torch.mean(m1_auprc_tensor)
    mean_m2_auroc = torch.mean(m2_auroc_tensor)
    mean_m2_auprc = torch.mean(m2_auprc_tensor)

    mean_t_ece = torch.mean(t_ece_tensor)
    mean_t_m1_auroc = torch.mean(t_m1_auroc_tensor)
    mean_t_m1_auprc = torch.mean(t_m1_auprc_tensor)
    mean_t_m2_auroc = torch.mean(t_m2_auroc_tensor)
    mean_t_m2_auprc = torch.mean(t_m2_auprc_tensor)

    std_accuracy = torch.std(accuracy_tensor) / math.sqrt(accuracy_tensor.shape[0])
    std_ece = torch.std(ece_tensor) / math.sqrt(ece_tensor.shape[0])
    std_m1_auroc = torch.std(m1_auroc_tensor) / math.sqrt(m1_auroc_tensor.shape[0])
    std_m1_auprc = torch.std(m1_auprc_tensor) / math.sqrt(m1_auprc_tensor.shape[0])
    std_m2_auroc = torch.std(m2_auroc_tensor) / math.sqrt(m2_auroc_tensor.shape[0])
    std_m2_auprc = torch.std(m2_auprc_tensor) / math.sqrt(m2_auprc_tensor.shape[0])

    std_t_ece = torch.std(t_ece_tensor) / math.sqrt(t_ece_tensor.shape[0])
    std_t_m1_auroc = torch.std(t_m1_auroc_tensor) / math.sqrt(t_m1_auroc_tensor.shape[0])
    std_t_m1_auprc = torch.std(t_m1_auprc_tensor) / math.sqrt(t_m1_auprc_tensor.shape[0])
    std_t_m2_auroc = torch.std(t_m2_auroc_tensor) / math.sqrt(t_m2_auroc_tensor.shape[0])
    std_t_m2_auprc = torch.std(t_m2_auprc_tensor) / math.sqrt(t_m2_auprc_tensor.shape[0])

    # ============================== #
    
    res_dict = {}
    res_dict["mean"] = {}
    res_dict["mean"]["accuracy"] = mean_accuracy.item()
    res_dict["mean"]["ece"] = mean_ece.item()
    res_dict["mean"]["m1_auroc"] = mean_m1_auroc.item()
    res_dict["mean"]["m1_auprc"] = mean_m1_auprc.item()
    res_dict["mean"]["m2_auroc"] = mean_m2_auroc.item()
    res_dict["mean"]["m2_auprc"] = mean_m2_auprc.item()
    res_dict["mean"]["t_ece"] = mean_t_ece.item()
    res_dict["mean"]["t_m1_auroc"] = mean_t_m1_auroc.item()
    res_dict["mean"]["t_m1_auprc"] = mean_t_m1_auprc.item()
    res_dict["mean"]["t_m2_auroc"] = mean_t_m2_auroc.item()
    res_dict["mean"]["t_m2_auprc"] = mean_t_m2_auprc.item()

    res_dict["std"] = {}
    res_dict["std"]["accuracy"] = std_accuracy.item()
    res_dict["std"]["ece"] = std_ece.item()
    res_dict["std"]["m1_auroc"] = std_m1_auroc.item()
    res_dict["std"]["m1_auprc"] = std_m1_auprc.item()
    res_dict["std"]["m2_auroc"] = std_m2_auroc.item()
    res_dict["std"]["m2_auprc"] = std_m2_auprc.item()
    res_dict["std"]["t_ece"] = std_t_ece.item()
    res_dict["std"]["t_m1_auroc"] = std_t_m1_auroc.item()
    res_dict["std"]["t_m1_auprc"] = std_t_m1_auprc.item()
    res_dict["std"]["t_m2_auroc"] = std_t_m2_auroc.item()
    res_dict["std"]["t_m2_auprc"] = std_t_m2_auprc.item()

    res_dict["mean"] = {}
    res_dict["mean"]["accuracy"] = mean_accuracy.item()
    res_dict["mean"]["ece"] = mean_ece.item()
    res_dict["mean"]["m1_auroc"] = mean_m1_auroc.item()
    res_dict["mean"]["m1_auprc"] = mean_m1_auprc.item()
    res_dict["mean"]["m2_auroc"] = mean_m2_auroc.item()
    res_dict["mean"]["m2_auprc"] = mean_m2_auprc.item()
    res_dict["mean"]["t_ece"] = mean_t_ece.item()
    res_dict["mean"]["t_m1_auroc"] = mean_t_m1_auroc.item()
    res_dict["mean"]["t_m1_auprc"] = mean_t_m1_auprc.item()
    res_dict["mean"]["t_m2_auroc"] = mean_t_m2_auroc.item()
    res_dict["mean"]["t_m2_auprc"] = mean_t_m2_auprc.item()

    res_dict["values"] = {}
    res_dict["values"]["accuracy"] = accuracies
    res_dict["values"]["ece"] = eces
    res_dict["values"]["m1_auroc"] = m1_aurocs
    res_dict["values"]["m1_auprc"] = m1_auprcs
    res_dict["values"]["m2_auroc"] = m2_aurocs
    res_dict["values"]["m2_auprc"] = m2_auprcs
    res_dict["values"]["t_ece"] = t_eces
    res_dict["values"]["t_m1_auroc"] = t_m1_aurocs
    res_dict["values"]["t_m1_auprc"] = t_m1_auprcs
    res_dict["values"]["t_m2_auroc"] = t_m2_aurocs
    res_dict["values"]["t_m2_auprc"] = t_m2_auprcs

    res_dict["info"] = vars(args)

    with open("res_"+str(model_name)+str(seed)+"_"+args.model_type+".json", "w",) as f:
        json.dump(res_dict, f)
