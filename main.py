from __future__ import print_function, division
import os
#import torch
import torchvision
#import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import *
from configs import *

configs = Configs

def loadCifa100():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0)
    ])
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    validation_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return [train_loader,val_loader,train_set, validation_set]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def getAccuracy(model,validationLoader,validationSet):
    top1 = 0.0
    top5 = 0.0
    for i, (inputs, labels) in enumerate(validationLoader):
        inputs, labels = (Variable(inputs.cuda()), Variable(labels.cuda()))
        outputs = model(inputs)
        outputs, labels = outputs.data, labels.data
        _, preds = outputs.topk(5, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(labels.view(1, -1).expand_as(preds))
        top1 += corrects[:1].view(-1).float().sum(0)
        top5 += corrects[:5].view(-1).float().sum(0)
    top1 = top1.mul_(100/len(validationSet) )
    top5 = top5.mul_(100/len(validationSet) )
    return [top1,top5]

def main():

    print('Dataset is loading ...........')
    train_loader, val_loader, train_set, validation_set = loadCifa100()
    print('Make checkpoint folder')
    checkpoint = os.path.join(configs.checkpoint, +configs.model+'_'+ configs.attention)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    model_path = os.path.join(checkpoint,configs.attention+'_'+'best_model.pt')
    print('Load model')
    model = get_model(configs.model, configs.norm,configs.attention)
    print('\tModel loaded: ' + configs.model )
    print('\tAttention type: ' + configs.attention )
    print("\tNumber of parameters: ", sum([param.nelement() for param in model.parameters()]))

    if configs.test:
        print("Run model in test mode")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            raise Exception('Cannot find model', model_path)

    if configs.gpu:
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.cuda()
        cudnn.benchmark = True

    if configs.test:
        print('Testing...')
        model.eval()
        top1, top5 = getAccuracy(model,val_loader,validation_set)
        print('Accuracy on Top 1 accuracy: %.2f' % top1)
        print('Accuracy on Top 5 accuracy: %.2f' % top5)
        return

    # Tensor board
    tb = SummaryWriter(checkpoint)

    # Optimization
    optimizer = optim.SGD(model.parameters(), lr=configs.lr, momentum=0.9, weight_decay=configs.weight_decay,nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, configs.schedule, gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1
    for epoch in range(args.num_epochs):
        # Train process
        learning_rate = optimizer.param_groups[0]['lr']
        print('Start training epoch {}. Learning rate {}'.format(epoch, learning_rate))
        model.train()
        num_batches = len(train_set) // configs.batch_size
        running_loss = 0
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            if configs.gpu:
                inputs, labels = (Variable(inputs.cuda()),Variable(labels.cuda()))
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.data.item()
            loss.backward()
            optimizer.step()
            del inputs, labels
        scheduler.step()
        train_loss = running_loss / num_batches
        print('\tTraining loss %f' % train_loss)

        model.eval()
        val_acc = 0
        num_batches = len(validation_set) // configs.batch_size + 1
        running_loss = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                if configs.gpu:
                    inputs, labels = (Variable(inputs.cuda()),Variable(labels.cuda()))
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.data.item()
                outputs, labels = outputs.data, labels.data
                _, preds = outputs.topk(1, 1, True, True)
                preds = preds.t()
                corrects = preds.eq(labels.view(1, -1).expand_as(preds))
                val_acc += torch.sum(corrects)
                del inputs, labels
        val_acc = val_acc.item() / len(validation_set) * 100
        val_loss = running_loss / num_batches
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        print('\tValidation loss %f' % (running_loss / num_batches))
        print('\tValidation acc', val_acc)
        print()

        # update tensorboard
        tb.add_scalar('Learning rate', learning_rate, epoch)
        tb.add_scalar('Train loss', train_loss, epoch)
        tb.add_scalar('Val loss', val_loss, epoch)
        tb.add_scalar('Val top1 acc', val_acc, epoch)

    print('Best validation acc %.2f' % best_val_acc)

if __name__ == '__main__':
    main()
