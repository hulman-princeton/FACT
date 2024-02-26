import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *

import sklearn.metrics
from sklearn.metrics import roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
    parser.add_argument("--config", default=None, help="Experiment configs")
    parser.add_argument("--tf_logger", type=ast.literal_eval, default=False, help="If true will save tensorboard compatible logs")
    parser.add_argument("--ckpt", default=None, help="The directory to models")

    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class Evaluator:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # networks
        self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(device)
        self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(device)

        # dataloaders
        self.test_loader = get_test_all_loader(args=self.args, config=self.config)

    def do_eval(self, loader):
        correct = 0
        preds_list = torch.tensor(()).to(self.device)
        labels_list = torch.tensor(()).to(self.device)
        softmax_list = torch.tensor(()).to(self.device)
        
        for it, (batch, domain) in enumerate(loader):
            data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
            features = self.encoder(data)
            scores = self.classifier(features)
            num_corr, pred, softmax = calculate_correct(scores, labels)
            correct += num_corr
            
            softmax_list = torch.cat((softmax_list, softmax.to(self.device)), 0)
            labels_list = torch.cat((labels_list, labels.to(self.device)), 0)
            preds_list = torch.cat((preds_list, pred.to(self.device)), 0)
            
        return correct, preds_list, softmax_list, labels_list

    def do_testing(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)

        self.encoder.eval()
        self.classifier.eval()
        if self.args.ckpt is not None:
            state_dict = torch.load(self.args.ckpt, map_location=lambda storage, loc: storage)
            encoder_state = state_dict["encoder_state_dict"]
            classifier_state = state_dict["classifier_state_dict"]
            self.encoder.load_state_dict(encoder_state)
            self.classifier.load_state_dict(classifier_state)

        acc_dict = {}    
        tp_dict = {}
        tn_dict = {}
        fp_dict = {}
        fn_dict = {}
        scores_dict = {}
        
        for key,val in self.test_loader.items():
            name = key
            loader = val
            with torch.no_grad():
                total = len(loader.dataset)
                correct, preds, softmax, labels = self.do_eval(loader)
                class_correct = correct
                class_acc = float(class_correct) / total
                self.logger.log_test(f'Test accuracy', {'class': class_acc})
                
                tp = 0
                tn = 0
                fp = 0
                fn = 0
        
                for i in range(len(preds)):
                    # correct class
                    if preds[i] == labels[i]:
                        if preds[i] == 1 and labels[i] == 1: tp += 1
                        elif preds[i] == 0 and labels[i] == 0: tn += 1
                        # ensure all predictions are sorted
                        else: print("equal not true")
        
                    # incorrect class
                    if preds[i] != labels[i]:
                        if preds[i] == 1 and labels[i] == 0: fp += 1
                        elif preds[i] == 0 and labels[i] == 1: fn += 1
                        # ensure all predictions are sorted
                        else: print("unequal not true")
        
                assert total == tp + tn + fp + fn
                
                labels = labels.cpu()
                softmax = softmax.cpu()
                
                score = roc_auc_score(labels, softmax[:,1])
                
                acc_dict[name] = class_acc
                tp_dict[name] = tp
                tn_dict[name] = tn
                fp_dict[name] = fp
                fn_dict[name] = fn
                scores_dict[name] = score

        return acc_dict, tp_dict, tn_dict, fp_dict, fn_dict, scores_dict


def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(args, config, device)
    acc_dict, tp_dict, tn_dict, fp_dict, fn_dict, auc_dict = evaluator.do_testing()

    for key in acc_dict:
        print("Loading for: ", key)
        print("Accuracy: ", acc_dict[key])
        print("AUC: ", auc_dict[key])
        print("TP: ", tp_dict[key])
        print("TN: ", tn_dict[key])
        print("FP: ", fp_dict[key])
        print("FN: ", fn_dict[key])

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
