"""
Script to evaluate a single model. 
"""

import os
import json
import math
import torch
import argparse
import torch.backends.cudnn as cudnn

# Importing the network models
from resnet_cNfc import resnet18

# Import metrics to compute
from metrics.classification_metrics import test_classification_net, test_classification_net_logits, test_classification_net_ensemble
from metrics.calibration_metrics import expected_calibration_error
from metrics.uncertainty_confidence import entropy, logsumexp
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble

# Import GMM utils
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit
from utils.ensemble_utils import load_ensemble, ensemble_forward_pass
from utils.eval_utils import model_load_name
from utils.train_utils import model_save_name
from utils.args import eval_args

# Temperature scaling
from utils.temperature_scaling import ModelWithTemperature

# ========================================================================================== #

def evaluation_args():
    model = "resnet18"
    sn_coeff = 3.0
    runs = 5
    ensemble = 5
    model_type = "softmax"

    parser = argparse.ArgumentParser(description="Training for calibration.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--seed", type=int, dest="seed", required=True, help="Seed to use")
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=False)

    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-b", type=int, default=batch_size, dest="batch_size", help="Batch size")
    parser.add_argument("--model", type=str, default=model, dest="model", help="Model to train")
    parser.add_argument("--runs", type=int, default=runs, dest="runs", help="Number of models to aggregate over",)

    parser.add_argument("-sn", action="store_true", dest="sn", help="whether to use spectral normalisation during training",)
    parser.set_defaults(sn=False)
    parser.add_argument("--coeff", type=float, default=sn_coeff, dest="coeff", help="Coeff parameter for spectral normalisation",)
    parser.add_argument("-mod", action="store_true", dest="mod", help="whether to use architectural modifications during training",)
    parser.set_defaults(mod=False)
    parser.add_argument("--ensemble", type=int, default=ensemble, dest="ensemble", help="Number of models in ensemble")
    parser.add_argument("--model-type", type=str, default=model_type, choices=["softmax", "ensemble", "gmm"], dest="model_type", help="Type of model to load for evaluation.",)

    return parser

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

    test_loader = dataset_loader[args.dataset].get_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)
    ood_test_loader = dataset_loader[args.ood_dataset].get_test_loader(batch_size=args.batch_size, pin_memory=args.gpu)

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

    # topt = None # DEPRECATED

    for i in range(args.runs):
        print(f"========== Run {i+1} ==========")
        
        # Loading the model to evaluate
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(batch_size=args.batch_size, augment=args.data_aug, val_seed=(args.seed+i), val_size=0.1, pin_memory=args.gpu,)
        num_outputs = grid_size
        net = models[args.model](
            num_outputs=num_outputs,
            temp=1.0,)
        
        # Using the gpu
        if args.gpu:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
            
        net.load_state_dict(torch.load("resnet182_100.model"))
        net.eval()

        # ============================== #

        # Evaluating the model
        (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net(net, test_loader, device)
        # ece(expected calibration error) represents quality of aleatoric uncertainty. ece measures expectation of difference between accuracy and confidence
        ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)

        # Temperature scaling the trained model (Calibrating the classifier to adjust the confidence of prediction)
        temp_scaled_net = ModelWithTemperature(net)
        temp_scaled_net.set_temperature(val_loader)
        # topt = temp_scaled_net.temperature # DEPRECATED

        (t_conf_matrix, t_accuracy, t_labels_list, t_predictions, t_confidences,) = test_classification_net(temp_scaled_net, test_loader, device)
        # ece(expected calibration error) represents quality of aleatoric uncertainty.
        t_ece = expected_calibration_error(t_confidences, t_predictions, t_labels_list, num_bins=15)

        # ============================== #
        
        if (args.model_type == "gmm"): # gmm
            # Evaluating the GMM model
            print("GMM Model")
            embeddings, labels = get_embeddings(
                net,
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

    with open("res_"+model_save_name(args.model, args.sn, args.mod, args.coeff, args.seed)+"_"+args.model_type+"_"+args.dataset+"_"+args.ood_dataset+".json", "w",) as f:
        json.dump(res_dict, f)
