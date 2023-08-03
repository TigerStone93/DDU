"""
Code to perform temperature scaling. Adapted from https://github.com/gpleiss/temperature_scaling
"""
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

from utils.ece_loss_cNfc import ECELoss

# ========================================================================================== #

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, log=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = 1.0
        self.log = log

    def forward(self, input):
        logits = self.model(input)
        # Performing the temperature scaling on logits by expanding the temperature to match the size of logits
        temperature_scaled_logits = logits / self.temperature
        return temperature_scaled_logits

    def set_temperature(self, valid_loader, cross_validate="nll"):
        # Tuning the tempearature of model using the validation set with cross-validation on ECE or NLL
        self.cuda()
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # Collecting all the logits and labels for validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Setting the NLL and ECE criterion
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # Calculating the NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            print("Before temperature scaling - NLL: %.3f, ECE: %.3f" % (before_temperature_nll, before_temperature_ece))

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            self.cuda()
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if cross_validate == "ece":
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll
        self.cuda()

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            print("Optimal temperature: %.3f" % self.temperature)
            print("After temperature scaling - NLL: %.3f, ECE: %.3f" % (after_temperature_nll, after_temperature_ece))

        return self
        
        return self.set_temperature_logits(logits, labels, cross_validate=cross_validate)

    def set_temperature_logits(self, logits, labels, cross_validate="nll"):
        """
        Tune the tempearature of the model (using the validation set) with cross-validation on ECE or NLL
        """
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            print("Before temperature - NLL: %.3f, ECE: %.3f" % (before_temperature_nll, before_temperature_ece))

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.1
        for i in range(100):
            self.temperature = T
            self.cuda()
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.1

        if cross_validate == "ece":
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll
        self.cuda()

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            print("Optimal temperature: %.3f" % self.temperature)
            print("After temperature - NLL: %.3f, ECE: %.3f" % (after_temperature_nll, after_temperature_ece))

        return self

    def get_temperature(self):
        return self.temperature
