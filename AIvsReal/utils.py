import torch 
from torch import nn
import numpy as np
from timeit import default_timer as timer
from tqdm.auto import tqdm


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.4f} seconds")

def convert_labels_to_tensors(y:torch.Tensor,number_of_labels):
    y_labels = torch.zeros((len(y),number_of_labels)).to(y.device)
    for i in range(len(y)):
        y_labels[i][y[i]] = 1
    return y_labels

def train_single_epoch(model:nn.Module,dataLoader:torch.utils.data.DataLoader,loss_fn:nn.Module,
                optimizer:torch.optim.Optimizer,accuracy_fn,device:torch.device):
    
    train_loss,train_acc = 0,0
    
    model.train()

    for batch , (X,y) in enumerate(tqdm(dataLoader)):
        X,y = X.to(device),y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred,convert_labels_to_tensors(y,2))
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,y_pred=y_pred.squeeze().argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_acc /= len(dataLoader)
    train_loss /= len(dataLoader)
    print(f"Training_Loss: {train_loss} | Training_Acc: {train_acc} ")

def test_step(model:nn.Module,dataLoader:torch.utils.data.DataLoader,loss_fn:nn.Module,
                       accuracy_fn,device:torch.device):

    test_acc,test_loss = 0,0

    model.eval()

    with torch.inference_mode():
        for X_test,y_test in dataLoader:
            X_test,y_test = X_test.to(device),y_test.to(device)

            test_pred = model(X_test)

            test_loss += loss_fn(test_pred,convert_labels_to_tensors(y_test,2))
            test_acc += accuracy_fn(y_true = y_test,y_pred = test_pred.squeeze().argmax(dim=1))
            
        test_loss /= len(dataLoader)
        test_acc /= len(dataLoader)
        print(f"Test Loss: {test_loss} | Test Acc: {test_acc}")

