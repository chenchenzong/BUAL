import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100

import numpy as np
import os
import argparse

from tqdm import tqdm
import pickle

import logging

import random
import copy

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    
class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

        
# For Tiny-Imagenet
class data(Dataset):
    def __init__(self, type, transform, labels, image_names):
        self.type = type
        if type == 'train':
            i = 0
            self.data = []
            for label in labels:
                image = []
                for image_name in image_names[i]:
                    image_path = os.path.join('../tiny-imagenet-200/train', label, 'images', image_name) 
                    data = Image.open(image_path).convert("RGB")
                    image.append(np.asarray(data))
                    data.close()
                self.data.append(image)
                i = i + 1
            self.data = np.array(self.data)
            self.data = self.data.reshape(-1, 64, 64, 3)
            self.uq_idxs = range(self.data.shape[0])
            self.targets = [i//500 for i in self.uq_idxs]
            self.uq_idxs = np.asarray(self.uq_idxs)
        elif type == 'val':
            self.data = []
            for image in image_names:
                image_path = os.path.join('../tiny-imagenet-200/val/images', image)
                data = Image.open(image_path).convert("RGB")
                self.data.append(np.asarray(data))
                data.close()
            self.data = np.array(self.data)
            self.uq_idxs = range(self.data.shape[0])
            self.targets = labels
            self.uq_idxs = np.asarray(self.uq_idxs)
        self.ToPILImage = transforms.ToPILImage()
        self.transform = transform
        
    def __getitem__(self, index):
        label = []
        image = []
        if self.type == 'train':
            label = self.targets[index]
            image = self.data[index]
            image = self.ToPILImage(np.uint8(image))
        if self.type == 'val':
            label = self.targets[index]
            image = self.data[index]
            image = self.ToPILImage(np.uint8(image))
        return self.transform(image), label, index
        
    def __len__(self):
        len = 0
        if self.type == 'train':
            len = self.data.shape[0]
        if self.type == 'val':
            len = self.data.shape[0]
        return len

def subsample_dataset(dataset, idxs):

    dataset.data = dataset.data[idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset

def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset    
    
    
def get_class_splits(dataset, num, split_idx=0):

    if dataset in ('cifar10'):
        train_classes = list(range(num))
        open_set_classes = [x for x in range(10) if x not in train_classes]

    elif dataset in ('cifar100'):
        train_classes = list(range(num))
        open_set_classes = [x for x in range(100) if x not in train_classes]
    
    elif dataset in ('tinyimagenet'):
        train_classes = list(range(num))
        open_set_classes = [x for x in range(200) if x not in train_classes]
    else:

        raise NotImplementedError

    return train_classes, open_set_classes    

    
def get_data_split(dataset, init_ratio, num, split_idx=0):
    if dataset == 'cifar10':
        num_classes = 10
        
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        trainset = CustomCIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
        testset = CustomCIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
        
        P_trainset = copy.deepcopy(trainset)
        P_testset = copy.deepcopy(testset)
        U_trainset = copy.deepcopy(trainset)
        U_testset = copy.deepcopy(testset)
        
        train_classes, open_set_classes = get_class_splits(dataset, num, split_idx)
        train_known = subsample_classes(P_trainset, include_classes=train_classes)
        test_known = subsample_classes(P_testset, include_classes=train_classes)
        train_unknown = subsample_classes(U_trainset, include_classes=open_set_classes)
        test_unknown = subsample_classes(U_testset, include_classes=open_set_classes)

        datasets = {
            'train':trainset,
            'train_known': train_known,
            'test_known': test_known, 
            'train_unknown': train_unknown,
            'test_unknown': test_unknown, 
        }
        
        known_ind = train_known.uq_idxs.tolist()
        labeled_ind = random.sample(known_ind, int(len(known_ind)*init_ratio))
        unlabeled_ind = [x for x in known_ind if x not in labeled_ind]
        openset_ind = train_unknown.uq_idxs.tolist()
        
        
        
    elif dataset == 'cifar100':
        num_classes = 100
        
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        trainset = CustomCIFAR100(root='./data', train=True,
                                        download=True, transform=train_transform)
        testset = CustomCIFAR100(root='./data', train=False,
                                       download=True, transform=test_transform)
        P_trainset = copy.deepcopy(trainset)
        P_testset = copy.deepcopy(testset)
        U_trainset = copy.deepcopy(trainset)
        U_testset = copy.deepcopy(testset)
        
        train_classes, open_set_classes = get_class_splits(dataset, num, split_idx)
        train_known = subsample_classes(P_trainset, include_classes=train_classes)
        test_known = subsample_classes(P_testset, include_classes=train_classes)
        train_unknown = subsample_classes(U_trainset, include_classes=open_set_classes)
        test_unknown = subsample_classes(U_testset, include_classes=open_set_classes)

        datasets = {
            'train':trainset,
            'train_known': train_known,
            'test_known': test_known, 
            'train_unknown': train_unknown,
            'test_unknown': test_unknown, 
        }
        
        known_ind = train_known.uq_idxs.tolist()
        labeled_ind = random.sample(known_ind, int(len(known_ind)*init_ratio))
        unlabeled_ind = [x for x in known_ind if x not in labeled_ind]
        openset_ind = train_unknown.uq_idxs.tolist()
    
    elif dataset == 'tinyimagenet':
        num_classes = 200
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        

        labels_t = []
        image_names = []
        with open('../tiny-imagenet-200/wnids.txt') as wnid:
            for line in wnid:
                labels_t.append(line.strip('\n'))
        for label in labels_t:
            txt_path = '../tiny-imagenet-200/train/'+label+'/'+label+'_boxes.txt'
            image_name = []
            with open(txt_path) as txt:
                for line in txt:
                    image_name.append(line.strip('\n').split('\t')[0])
            image_names.append(image_name)
        labels = np.arange(200)

        val_labels_t = []
        val_labels = []
        val_names = []
        with open('../tiny-imagenet-200/val/val_annotations.txt') as txt:
            for line in txt:
                val_names.append(line.strip('\n').split('\t')[0])
                val_labels_t.append(line.strip('\n').split('\t')[1])
        for i in range(len(val_labels_t)):
            for i_t in range(len(labels_t)):
                if val_labels_t[i] == labels_t[i_t]:
                    val_labels.append(i_t)
        val_labels = np.array(val_labels)
        
        trainset = data("train", train_transform,labels_t,image_names)
        
        testset = data("val", test_transform,val_labels,val_names)
        
        P_trainset = copy.deepcopy(trainset)
        P_testset = copy.deepcopy(testset)
        U_trainset = copy.deepcopy(trainset)
        
        train_classes, open_set_classes = get_class_splits(dataset, num, split_idx)
        train_known = subsample_classes(P_trainset, include_classes=train_classes)
        test_known = subsample_classes(P_testset, include_classes=train_classes)
        train_unknown = subsample_classes(U_trainset, include_classes=open_set_classes)

        datasets = {
            'train':trainset,
            'test_known':test_known
        }
        
        known_ind = train_known.uq_idxs.tolist()
        labeled_ind = random.sample(known_ind, int(len(known_ind)*init_ratio))
        unlabeled_ind = [x for x in known_ind if x not in labeled_ind]
        openset_ind = train_unknown.uq_idxs.tolist()
        
    return datasets, labeled_ind, unlabeled_ind, openset_ind, train_classes

def Aux_train(net, trainloader, openset_ind, optimizer, criterion, times, epoch, logger, device = 'cuda'):
    print('\nEpoch: %d' % epoch)
    net.train()
    correct = 0
    total = 0
    loader_bar = tqdm(trainloader)
    for inputs, targets, idx in loader_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, _ = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loader_bar.set_description('Train epoch %d, Acc: %.3f%% (%d/%d)'
                     % (epoch, 100.*correct/(total+1e-4), correct, total))
    logger.info('Train epoch %d, Acc: %.3f%% (%d/%d)'
                     % (epoch, 100.*correct/(total+1e-4), correct, total))


def PL_train(net, trainloader, optimizer, criterion, num_classes, times, epoch, logger, device = 'cuda'):
    print('\nEpoch: %d' % epoch)
    net.train()
    correct = 0
    total = 0
    loader_bar = tqdm(trainloader)
    for inputs, targets, _, idx in loader_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, _ = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loader_bar.set_description('Train epoch %d, Acc: %.3f%% (%d/%d)'
                     % (epoch, 100.*correct/(total+1e-4), correct, total))
    logger.info('Train epoch %d, Acc: %.3f%% (%d/%d)'
                     % (epoch, 100.*correct/(total+1e-4), correct, total))

def NL_train(net, trainloader, optimizer, criterion_nll, num_classes, times, epoch, logger, device = 'cuda'):
    print('\nEpoch: %d' % epoch)
    net.train()
    ln_neg = 1
    correct = 0
    total = 0
    loader_bar = tqdm(trainloader)
    for inputs, targets, _, idx in loader_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        targets_neg = (targets.unsqueeze(-1).repeat(1, ln_neg) + torch.LongTensor(len(targets), ln_neg).random_(1, num_classes).to(device)) % num_classes

        optimizer.zero_grad()
        _,outputs = net(inputs)
        
        prob = F.softmax(outputs, -1)
        selNL = torch.ones_like(prob)
        if epoch > 500:
            selNL[prob < 0.2] = 0
                
        s_neg = torch.log(torch.clamp(1.-prob, min=1e-5, max=1.))
        s_neg = s_neg.expand(s_neg.size())
        
        loss_neg = criterion_nll(s_neg.repeat(ln_neg, 1)*selNL, targets_neg.t().contiguous().view(-1))
        loss_neg.backward()

        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loader_bar.set_description('Train epoch %d, Acc: %.3f%% (%d/%d)'
                     % (epoch, 100.*correct/(total+1e-4), correct, total))
    logger.info('Train epoch %d, Acc: %.3f%% (%d/%d)'
                     % (epoch, 100.*correct/(total+1e-4), correct, total))


def PL_test(net, testloader, criterion, best_acc, times, epoch, logger, device = 'cuda'):
    net.eval()
    loss_meter = AverageMeter('test_loss')
    correct = 0
    total = 0
    loader_bar = tqdm(testloader)
    with torch.no_grad():
        for inputs, targets, idx in loader_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_ = net(inputs)
            loss = criterion(outputs, targets)
            loss_meter.update(loss.mean().item(), targets.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loader_bar.set_description('Test epoch %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch, loss_meter.avg, 100.*correct/total, correct, total))
        logger.info('Test epoch %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch, loss_meter.avg, 100.*correct/total, correct, total))

    # Save checkpoint.
    
    acc = 100.*correct/total
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_' + str(times) +'-last.pth')
    best_acc = acc
     
    return best_acc

        
def test_NL(net, dataloader, times, epoch, device = 'cuda',threshold=0.6):
    net.eval()
    loader_bar = tqdm(dataloader)
    
    known_prob = [] 
    idx_list = []
    with torch.no_grad():
        for inputs, targets, idx in loader_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            outputs = F.softmax(outputs, -1)
            prob = outputs
            known_prob.extend(prob.cpu().numpy().tolist())
            idx_list.extend(idx.numpy().tolist())
    
    idx_list2 = np.asarray(idx_list)
    idx_list2 = idx_list2[np.argsort(idx_list)].tolist()
    print(idx_list2[:10])
    known_prob = np.asarray(known_prob)
    save_list = [idx_list2, known_prob[np.argsort(idx_list)].tolist()]
    return idx_list2, known_prob[np.argsort(idx_list)].tolist()

def test_PL(net, dataloader, times, epoch, device = 'cuda',threshold=0.6):
    net.eval()
    loader_bar = tqdm(dataloader)
    
    known_prob = [] 
    idx_list = []
    with torch.no_grad():
        for inputs, targets, idx in loader_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_ = net(inputs)
            outputs = F.softmax(outputs, -1)
            prob = outputs
            known_prob.extend(prob.cpu().numpy().tolist())
            idx_list.extend(idx.numpy().tolist())
    
    idx_list2 = np.asarray(idx_list)
    idx_list2 = idx_list2[np.argsort(idx_list)].tolist()
    print(idx_list2[:10])
    known_prob = np.asarray(known_prob)
    return idx_list2, known_prob[np.argsort(idx_list)].tolist()

def test_Aux(net, dataloader, times, epoch, device = 'cuda',threshold=0.6):
    net.eval()
    loader_bar = tqdm(dataloader)
    known_prob = [] 
    known_pred = [] 
    idx_list = []
    with torch.no_grad():
        for inputs, targets, idx in loader_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            outputs = F.softmax(outputs, -1)
            _, predicted = outputs.max(1)
            known_pred.extend(predicted.cpu().numpy().tolist())
            known_prob.extend(outputs.cpu().numpy().tolist())
            idx_list.extend(idx.numpy().tolist())
    
    idx_list2 = np.asarray(idx_list)
    idx_list2 = idx_list2[np.argsort(idx_list)].tolist()
    print(idx_list2[:10])
    known_prob = np.asarray(known_prob)
    known_pred = np.asarray(known_pred)
    save_list = [idx_list2, known_pred[np.argsort(idx_list)].tolist(), known_prob[np.argsort(idx_list)].tolist()]
    return idx_list2, known_pred[np.argsort(idx_list)].tolist(), known_prob[np.argsort(idx_list)].tolist()
