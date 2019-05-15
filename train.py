import csv
import os
import numpy as np
from argparse import ArgumentParser as ArgPar
from collections import namedtuple
import sys
import math
import traceback
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
from dataset import get_data
from tqdm import tqdm
from wide_resnet import WideResNet

def get_arguments():
    argp = ArgPar()
    hp = {"batch_size": 128,
           "lr": 1.0e-1,
           "momentum": 0.9,
           "weight_decay": 5.0e-4,
           "width_coef1": 10,
           "width_coef2": 10,
           "width_coef3": 10,
           "n_blocks1": 4,
           "n_blocks2": 4,
           "n_blocks3": 4,
           "drop_rates1": 0.3,
           "drop_rates2": 0.3,
           "drop_rates3": 0.3,
           "lr_decay": 0.2}

    for name, v in hp.items():
        argp.add_argument("-{}".format(name), type = type(v), default = v)
    
    parsed_parameters = argp.parse_args()
    HyperParameters = {}
    
    for k in hp.keys():
        HyperParameters[k] =  eval("parsed_parameters.{}".format(k))
        
    return HyperParameters

def accuracy(y, target):
    pred = y.data.max(1, keepdim = True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()

    return acc

def print_result(values):
    f1 = "  {"
    f2 = "}  "
    f_int = "{}:<20"
    f_float = "{}:<20.5f"

    f_vars = ""
    
    for i, v in enumerate(values):
        if type(v) == float:
            f_vars += f1 + f_float.format(i) + f2
        else:
            f_vars += f1 + f_int.format(i) + f2
    
    print(f_vars.format(*values))

def train(device, optimizer, learner, train_data, loss_func):
    train_acc, train_loss, n_train = 0, 0, 0
    lr = optimizer.param_groups[0]["lr"]
    learner.train()
    bar = tqdm(desc = "Training", total = len(train_data), leave = False)
    
    for data, target in train_data:
        data, target =  data.to(device), target.to(device) 
        y = learner(data)            
        loss = loss_func(y, target)
        optimizer.zero_grad() # clears the gradients of all optimized tensors.
        loss.backward() 
        optimizer.step() # renew learning rate

        train_acc += accuracy(y, target)

        train_loss += loss.item() * target.size(0)
        n_train += target.size(0)

        bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f}".format(train_loss / n_train, float(train_acc) / n_train))
        bar.update()
    bar.close()
    bar = tqdm(desc = "Testing", total = len(test_data), leave = False)

    return train_acc / n_train, train_loss / n_train

def test(device, optimizer, learner, test_data, loss_func):
    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        y = learner(data)
        loss = loss_func(y, target)

        test_acc += accuracy(y, target)
        test_loss += loss.item() * target.size(0)
        n_test += target.size(0)

        bar.update()
    bar.close()
    
    return test_acc / n_test, test_loss / n_test

def main(learner):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data, test_data = get_data(learner.batch_size)
    
    learner = learner.to(device)
    cudnn.benchmark = True

    optimizer = optim.SGD( \
                        learner.parameters(), \
                        lr = learner.lr, \
                        momentum = learner.momentum, \
                        weight_decay = learner.weight_decay, \
                        nesterov = True \
                        )
    
    loss_func = nn.CrossEntropyLoss().cuda()

    milestones = learner.lr_step
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = learner.lr_decay)


    rsl_keys = ["lr", "epoch", "TrainAcc", "TrainLoss", "TestAcc", "TestLoss", "Time"]
    rsl = []
    y_out = 1.0e+8
    
    print_result(rsl_keys)
    
    for epoch in range(learner.epochs):
        train_acc, train_loss = train(device, optimizer, learner, train_data, loss_func)     
        
        learner.eval() # switch to test mode( make model not save the record of calculation)

        with torch.no_grad():
            test_acc, test_loss = test(device, optimizer, learner, test_data, loss_func)


        time_now = str(datetime.datetime.today())
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_acc, train_loss, test_acc, test_loss, time_now])})
       
        y_out = min(y_out, test_loss)
        print_result(rsl[-1].values())
        scheduler.step()
        
if __name__ == "__main__":
    hp_dict = get_arguments()
    hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in hp_dict.keys() ) )
    hyperparameters = hp_tuple(**hp_dict)
    print(hp_dict)
    learner = WideResNet(hyperparameters)
    
    print("Start Training")
    print("")
    print("")
    main(learner)
