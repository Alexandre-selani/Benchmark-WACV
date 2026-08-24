"""
    Helper functions for training and evaluation.

    progress_bar and format_time function was taken from https://github.com/kuangliu/pytorch-cifar which mimics xlua.progress

    Dimity Miller, 2020
"""

import os
import sys
import time
import math
import numpy as np
import torch


try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except:
    term_width = 84

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def find_anchor_means(net, dataloader,device,num_classes):
    
    all_logits = []
    all_targets = []
    all_predicts = []
    for X,y in dataloader:
        X = X.to(device)

        net.skip_distances = True
        net.eval()

        with torch.no_grad():
            logits = net(X)
            _,predicts = torch.max(logits,1)
           
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
        all_predicts.append(predicts.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_predicts = torch.cat(all_predicts)

    means = torch.zeros(num_classes,num_classes,dtype=torch.float64)

    for cl in range(num_classes):
        mask = (all_targets == cl) & (all_predicts == cl)
        x = all_logits[mask]
        x = np.squeeze(x)
        
        means[cl] = torch.mean(x, dim = 0)

    #print(means)
    return means

def gather_outputs(net,dataloader,device):
    """Returns logits, anchor distances and targets."""
    all_logits = []
    all_targets = []
    all_distances = []

    for X,y in dataloader:
        X = X.to(device)
        y = y.to(device)

        net.eval()
        net.skip_distances = False

        with torch.no_grad():
            logits,distances = net(X)
            
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
        all_distances.append(distances.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_distances = torch.cat(all_distances)
    
    return all_logits,all_distances,all_targets


def SoftmaxTemp(logits, T = 1):
    num = torch.exp(logits/T) 
    denom = torch.sum(torch.exp(logits/T), 1).unsqueeze(1) 
    return num/denom
