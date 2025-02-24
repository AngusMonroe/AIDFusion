"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import csv
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from metrics import accuracy_TU as accuracy, precision, recall, f1, roc_auc, accuracy_all_classes

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, device)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_labels = batch_labels.to(device)
            batch_scores = model.forward(batch_graphs, device)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc


def evaluate_network_all_metric(model, device, data_loader, epoch=0, path=''):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    labels = []
    masks = []

    with torch.no_grad():
        y_pred = []
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_labels = batch_labels.to(device)
            batch_scores = model.forward(batch_graphs, device)
            loss = model.loss(batch_scores, batch_labels)

            labels.append(batch_labels.detach().cpu())
            y_pred.append(batch_scores.detach().cpu())

            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

        y_true = torch.cat(labels, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        if len(np.unique(y_true)) == 2:
            test_precision = precision(y_pred, y_true)
            test_recall = recall(y_pred, y_true)
            test_f1 = f1(y_pred, y_true)
            test_roc_auc = roc_auc(y_pred, y_true)
        else:
            test_precision = epoch_test_acc
            test_recall = epoch_test_acc
            test_f1 = epoch_test_acc
            test_roc_auc = epoch_test_acc

        all_acc = accuracy_all_classes(y_pred, y_true)

        if path and masks:
            # attn_map = torch.cat(masks, dim=0).cpu()
            torch.save(masks, path)

    return epoch_test_loss, epoch_test_acc, test_precision, test_recall, test_f1, test_roc_auc, all_acc


def check_patience(all_losses, best_loss, best_epoch, curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter
