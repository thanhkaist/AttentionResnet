from __future__ import print_function, division
from torch.utils.data import DataLoader
from models import *
from torch.autograd import Variable
import torch.nn as nn
import os.path as osp
import torch.optim as optim
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision
import torch.backends.cudnn as cudnn
import argparse
import os
import numpy as np
import glob
import pdb
import matplotlib.cm as cm

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

import cv2
from torchsummary import summary

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
SAMPLES = np.arange(5)

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def get_image(var_image):
    image = var_image.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = (image*STD + MEAN)*255
    return image


def get_image_links():
    list = glob.glob("results/sample/*.png")
    return list

def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    #  image size 32*32
    raw_image = cv2.resize(raw_image, (32,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

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
    val_loader = DataLoader(valset, batch_size=100,shuffle=False, num_workers=4, pin_memory=True)

    # Model

    checkpoint = os.path.join(args.checkpoint, args.model + "_" + args.attention)
    model_path = os.path.join(checkpoint, args.attention + '_' + 'best_model.pt')
    print('Loading model...')
    model = get_model(args.model,'bn',args.attention)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise Exception('Cannot find model', model_path)
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    print('\tModel loaded: ' + args.model)
    print('\tAttention type: ' + args.attention)
    print("\tNumber of parameters: ", sum([param.nelement() for param in model.parameters()]))

    result_path = os.path.join('results', args.model + "_" + args.attention)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if True:
        image_paths = get_image_links()
        images = []
        raw_images = []
        print("Images:")
        for i, image_path in enumerate(image_paths):
            print("\t#{}: {}".format(i, image_path))
            image, raw_image = preprocess(image_path)
            images.append(image)
            raw_images.append(raw_image)
        images = torch.stack(images).to("cuda")
    
    model.eval()
    # if False:
    #     summary(model, (3, 32, 32))
    #     return

    # Get sample for evaluate
    GET_SAMPLE = False
    if GET_SAMPLE:
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = (Variable(inputs.cuda()),
                              Variable(labels.cuda()))
            outputs = model(inputs)

            _, preds = outputs.topk(1, 1, True, True)
            for sample in SAMPLES:
                image = get_image(inputs[sample])
                save_image(result_path, image, sample, preds[sample], labels[sample])
            break


    print("Vanilla Backpropagation:")
    topk = 1
    target_layer = "layer4.2"
    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)

    for i in range(topk):
        # In this example, we specify the high confidence classes
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, LABELS[ids[j, i]], probs[j, i]))


    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, LABELS[ids[j, i]], probs[j, i]))


    deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, LABELS[ids[j, i]], probs[j, i]))

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    result_path,
                    "{}-gradcam-{}-{}.png".format(
                        j, target_layer, LABELS[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


    print('Finish!!!')

if __name__ == '__main__':
    main()
