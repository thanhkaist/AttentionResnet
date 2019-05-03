from __future__ import print_function, division
from torch.utils.data import DataLoader
from models import *
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision
import torch.backends.cudnn as cudnn
import argparse
import os
import numpy as np
from grad_cam import GradCAM
import cv2

args={}
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--attention', type=str, default='no')

args = parser.parse_args()


LABELS = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
MEAN = np.array([125.3, 123.0, 113.9]) / 255.0
STD = np.array([63.0, 62.1, 66.7]) / 255.0
SAMPLES = np.arange(20)

def save_gradcam(path, filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    filename = os.path.join(path, filename)
    cv2.imwrite(filename, np.uint8(gcam))

def get_image(var_image):
    image = var_image.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = (image*STD + MEAN)*255
    return image

def save_image(path, image, index, pred, label):
    file_name = '{}_{}_{}.png'.format(index, LABELS[pred], LABELS[label])
    file_name = os.path.join(path, file_name)
    cv2.imwrite(file_name, image)

def main():

    # Dataset
    print('Creating dataset...')
    transform_val= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
            ])
    valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    val_loader = DataLoader(valset, batch_size=100,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model

    checkpoint = os.path.join(args.checkpoint, args.model + "_" + args.attention)
    model_path = os.path.join(checkpoint, args.attention + '_' + 'best_model.pt')
    print('Loading model...')
    model = get_model(args.model)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise Exception('Cannot find model', model_path)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    print('\tModel loaded: ' + args.model)
    print('\tAttention type: ' + args.attention)
    print("\tNumber of parameters: ", sum([param.nelement() for param in model.parameters()]))

    # result
    result_path = os.path.join('results', args.model + "_" + args.attention)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    

    gcam = GradCAM(model=model)
    
    model.eval()
    acc = 0
    for i, (inputs, labels) in enumerate(val_loader):
        inputs, labels = (Variable(inputs.cuda()),
                          Variable(labels.cuda()))
        outputs = model(inputs)
        outputs, labels = outputs.data, labels.data
        _, preds = outputs.topk(1, 1, True, True)
        corrects = preds.eq(labels.view(-1, 1).expand_as(preds))
        acc += torch.sum(corrects)
        for sample in SAMPLES:
            image = get_image(inputs[sample])
            probs, idx = gcam.forward(inputs[sample].unsqueeze(0))
            gcam.backward(idx=idx[1])
            output = gcam.generate(target_layer='layer4.2')
            save_gradcam(result_path, '{}_gcam.png'.format(sample), output, image)
            save_image(result_path, image, sample, preds[sample], labels[sample])
        break
    acc = acc.item()/len(valset)*100
    print('Finish!!!')

if __name__ == '__main__':
    main()
