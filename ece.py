import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.patches as mpatches

def T_scaling(logits, args):
  temperature = args.get('temperature', None)
  return torch.div(logits, temperature)
def calc_bins(preds,labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes
def get_metrics(preds,label_oneh):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds,label_oneh)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE


def draw_reliability_graph(preds,path,label_oneh):
    ECE, MCE = get_metrics(preds,label_oneh)
    bins, _, bin_accs, _, _ = calc_bins(preds,label_oneh)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE * 100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE * 100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    # plt.show()

    plt.savefig(path, bbox_inches='tight')

warnings.filterwarnings("ignore")

plt.ion()




def test(net,test_loader,device):
    preds = []
    labels_oneh = []
    correct = 0
    net.eval()

    with torch.no_grad():
        # for index,（data, targets） in tqdm(enumerate(test_loader), total=len(test_loader), leave = True) :
        #     for data, targets in tqdm(train_loader):
        #         data = data.to(device=device)
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        for index, (images, labels) in loop:
            print(len(loop))
            images, labels = torch.tensor(images), torch.tensor(labels)
            # images, labels = Variable(images), Variable(labels)
            print(index)
            images=images.float()
            # images=torch.tensor(images,dtype=torch.float32)
            # labels = torch.tensor(labels, dtype=torch.int64)

            images=images.to(device)
            labels=labels.to(device)
            pred = net(images)

            # if calibration_method:
            #     pred = calibration_method(pred, kwargs)

            # Get softmax values for net input and resulting class predictions
            sm = nn.Softmax(dim=1)
            pred = sm(pred)

            _, predicted_cl = torch.max(pred.data, 1)
            pred = pred.cpu().detach().numpy()

            # Convert labels to one hot encoding
            label_oneh = torch.nn.functional.one_hot(labels, num_classes=2)
            label_oneh = label_oneh.cpu().detach().numpy()

            preds.extend(pred)
            labels_oneh.extend(label_oneh)

            # Count correctly classified samples for accuracy
            correct += sum(predicted_cl == labels).item()

            loop.set_description(f'All [{index}/{(loop)}]')
            loop.set_postfix( acc=(sum(predicted_cl == labels).item())/100)

    preds = np.array(preds).flatten()
    labels_oneh = np.array(labels_oneh).flatten()

    correct_perc = correct / len(test_loader)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_perc))
    print(correct_perc)

    return preds, labels_oneh

