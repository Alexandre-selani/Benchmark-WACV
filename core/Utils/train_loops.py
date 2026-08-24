import torch
import numpy as np

from Utils.device import DEVICE as device

def train(train_loader,model,criterion,optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss/(batch_idx+1), correct/total

def eval(val_loader,model,criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return val_loss/(batch_idx+1), correct/total


def train_cac(train_loader,model,criterion,optimizer,num_classes,lbda):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        cac_loss,anchor_loss,tuplet_loss = criterion(outputs[1], targets,num_classes,lbda)
        
        cac_loss.backward()
        optimizer.step()

        train_loss += cac_loss.item()
        _, predicted = outputs[1].min(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss/(batch_idx+1), correct/total

def eval_cac(val_loader,model,criterion,num_classes,lbda):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            cac_loss,anchor_loss,tuplet_loss = criterion(outputs[1], targets,num_classes,lbda)

        val_loss += cac_loss.item()
        _, predicted = outputs[1].min(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return val_loss/(batch_idx+1), correct/total

def predict(test_loader,model):
    model.eval()
    y_pred = []
    scores = []
    outputs =[]
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets

        
        with torch.no_grad():
            output = model(inputs)
            score, predicted = output.max(1)

        y_pred.append(predicted)
        scores.append(score)
        outputs.append(output)

    y_pred = torch.concat(y_pred)
    scores = torch.concat(scores)
    outputs = torch.concat(outputs)
    return y_pred,scores,outputs