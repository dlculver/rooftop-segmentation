import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score


def compute_metrics(pred, target):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)
    iou = jaccard_score(target, pred)

    return accuracy, precision, recall, f1, iou

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if outputs.size() != labels.size():
                    outputs = F.interpolate(outputs, size=labels.size()[2:], mode='bilinear', align_corners=True)
            loss = criterion(outputs, labels)
            dice = dice_loss(torch.sigmoid(outputs), labels)

            running_loss += loss.item() * inputs.size(0)
            running_dice_loss += dice.item() * inputs.size(0)

            preds = outputs > 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    total_loss = running_loss / len(dataloader.dataset)
    total_dice_loss = running_dice_loss / len(dataloader.dataset)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy, precision, recall, f1, iou = compute_metrics(all_preds, all_labels)

    output_dict = {
        'total_loss': total_loss, 
        'total_dice_loss': total_dice_loss,
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1, 
        'iou': iou
    }

    return output_dict

