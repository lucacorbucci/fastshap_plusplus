import warnings

import numpy as np
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

warnings.filterwarnings("ignore")


class Learning:
    def test(model, test_loader, device):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        losses = []
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        z_list = []

        with torch.no_grad():
            for _, (data, z, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.long()
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                total += target.size(0)
                losses.append(loss.item())

                y_pred.extend(pred)
                y_true.extend(target)
                z_list.extend(z)

        test_loss = np.mean(losses)
        accuracy = correct / total

        y_true = [item.item() for item in y_true]
        y_pred = [item.item() for item in y_pred]
        f1score = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")

        return {
            "Accuracy": accuracy,
            "Loss": test_loss,
            "F1 Score": f1score,
            "Precision": precision,
            "Recall": recall,
            "y_pred": y_pred,
            "z_list": z_list,
            "true_y": y_true,
            "y_pred": y_pred,
            "z_list": z_list,
        }

    def train(model, train_loader, optimizer, device):
        model.train()
        criterion = nn.CrossEntropyLoss(reduction="none")

        losses = []
        total_correct = 0
        total = 0
        y_pred = []
        y_true = []
        z_list = []

        MAX_PHYSICAL_BATCH_SIZE = 128
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for _, (data, z, target) in enumerate(memory_safe_data_loader):
                optimizer.zero_grad()
                data = data.to(device)
                target = target.long()
                target = target.to(device)

                # compute output
                outputs = model(data)
                loss = criterion(outputs, target).mean()

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == target).float().sum()
                total_correct += correct
                total += target.size(0)

                losses.append(loss.item())

                loss.backward()
                optimizer.step()
                y_pred.extend(predicted)
                y_true.extend(target)
                z_list.extend(z)

        y_true = [item.item() for item in y_true]
        y_pred = [item.item() for item in y_pred]
        train_accuracy = total_correct / total
        train_loss = np.mean(losses)
        f1score = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")

        return {
            "Train Accuracy": train_accuracy,
            "Train Loss": train_loss,
            "F1 Score": f1score,
            "Precision": precision,
            "Recall": recall,
            "y_pred": y_pred,
            "z_list": z_list,
        }
