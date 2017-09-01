#!/usr/bin/env python

"""
    forward.py
"""

import os
import sys
import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import functional as F

import torchvision.models as models
from torchvision import transforms

from PIL import Image

# --
# IO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--classify', action='store_true')
    return parser.parse_args()


def partial_forward(model, x):
    """ resnet forward pass w/o final fc layer """
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    
    x = model.avgpool(x)
    x = x.view(x.size(0), -1)
    return x


if __name__ == "__main__":
    
    args = parse_args()
    
    model = torch.load('./whole_%s_places365.pth.tar' % args.arch).cuda()
    _ = model.eval()
    
    print >> sys.stderr, model
    
    prep = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    class_names = np.array([line.strip().split(' ')[0][3:] for line in open('categories_places365.txt')])
    
    for path in sys.stdin:
        path = path.strip()
        img = Image.open(path)
        img = Variable(prep(img).unsqueeze(0), volatile=True).cuda()
        
        if args.classify:
            logit = model.forward(img)
            h_x = F.softmax(logit).data.squeeze()
            probs, idx = h_x.sort(0, True)
            
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()
            classes = class_names[idx]
            
            print dict(zip(classes, probs))
        else:
            logit = partial_forward(model, img)
            logit = logit.data.cpu().numpy().squeeze()
            print '\t'.join([path] + map(str, logit))


