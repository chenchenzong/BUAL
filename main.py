import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import argparse

from utils import *
from dataset import NL_Dataset, PL_Dataset

from resnet import *
from tqdm import tqdm
import pickle
import copy

import logging

parser = argparse.ArgumentParser(description='BU Active Learning')
parser.add_argument('--dataset', type=str, default='cifar100', help="")
parser.add_argument('--image_size', type=int, default=32)

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

parser.add_argument('--init_ratio', default=0.08, type=int, help='The initial ratio of labeled data')

parser.add_argument('--type', type=int, default=1, help='Query strategies select:[1:B-LC,2:B-Margin,3:B-Entropy]')
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--total_times', type=int, default=8)
parser.add_argument('--aux_epochs', type=int, default=100)
parser.add_argument('--pl_epochs', type=int, default=100)
parser.add_argument('--nl_epochs', type=int, default=100)
parser.add_argument('--known_classes', type=int, default=4)

parser.add_argument('--query_size', type=int, default=1500)

parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

output_file_name = 'output_' + args.dataset + '_' + str(args.known_classes) + '_seed' + str(args.seed)

logging.basicConfig(level=logging.INFO,
                    filename=args.dataset+'_'+str(args.known_classes)+'_seed'+str(args.seed)+'.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
datasets, labeled_ind, unlabeled_ind, openset_ind, train_classes = get_data_split(args.dataset, args.init_ratio, args.known_classes) 
testloader = DataLoader(datasets['test_known'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print(len(set(labeled_ind)), len(set(unlabeled_ind)), len(set(openset_ind)))

criterion = nn.CrossEntropyLoss()
criterion_nll = nn.NLLLoss()

known_openset_ind = []

os.mkdir(output_file_name)
os.chdir(output_file_name)

ratio = 0

for times in range(args.total_times):
    with open("label_ind_" + str(times) +".pkl", 'wb') as f:
        pickle.dump(labeled_ind, f)
    with open("unlabel_ind_" + str(times) +".pkl", 'wb') as f:
        pickle.dump(unlabeled_ind, f)
    with open("openset_ind_" + str(times) +".pkl", 'wb') as f:
        pickle.dump(openset_ind, f)
    with open("known_openset_ind_" + str(times) +".pkl", 'wb') as f:
        pickle.dump(known_openset_ind, f)
    
    #Aux_train

    if known_openset_ind != []:
        Aux_ind = []
        Aux_ind.extend(labeled_ind)
        Aux_ind.extend(known_openset_ind)
        Aux_datasets = copy.deepcopy(datasets['train'])
        for ind in known_openset_ind:
            Aux_datasets.targets[ind] = args.known_classes
        Aux_loader = DataLoader(Aux_datasets, batch_size=100, sampler=SubsetRandomSampler(Aux_ind))
        Aux_net = ResNet18(len(train_classes)+1)
        Aux_net = Aux_net.to(device)
        Aux_optimizer = optim.SGD(Aux_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        for epoch in range(args.aux_epochs):
            Aux_train(Aux_net, Aux_loader, known_openset_ind, Aux_optimizer, criterion, times, epoch, logger, device)
    
        query_pool_ind = []
        query_pool_ind.extend(unlabeled_ind)
        query_pool_ind.extend(openset_ind)
        query_loader = DataLoader(datasets['train'], batch_size=100, sampler=SubsetRandomSampler(query_pool_ind))
    
        Aux_ind, Aux_pred, Aux_prob = test_Aux(Aux_net, query_loader, times, epoch, device)
        Aux_prob = np.asarray(Aux_prob)[:,-1]
        pred_openset = []
        Aux_prob2 = []
        for i in range(len(Aux_ind)):
            if Aux_pred[i] == args.known_classes:
                pred_openset.append(Aux_ind[i])
            else:
                Aux_prob2.append(Aux_prob[i])
        

    # PL
    PL_dataset = PL_Dataset(datasets, labeled_ind, unlabeled_ind, openset_ind, train_classes)
    
    PL_loader = DataLoader(PL_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(PL_dataset.get_train_ind()))
    
    net = ResNet18(len(train_classes))
    net = net.to(device)
    PL_optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    best_acc = 0
    
    for epoch in range(args.pl_epochs):
        PL_train(net, PL_loader, PL_optimizer, criterion, len(train_classes), times, epoch, logger, device)
        best_acc = PL_test(net, testloader, criterion, best_acc, times, epoch, logger, device)
      
    # NL
    
    labeled_loader = DataLoader(datasets['train'], batch_size=100, sampler=SubsetRandomSampler(PL_dataset.labeled_ind))
    unlabeled_loader = DataLoader(datasets['train'], batch_size=100, sampler=SubsetRandomSampler(PL_dataset.sel_unlabeled_ind))
    openset_loader = DataLoader(datasets['train'], batch_size=100, sampler=SubsetRandomSampler(PL_dataset.sel_openset_ind))
    
    NL_dataset = NL_Dataset(datasets, labeled_ind, PL_dataset.target1, PL_dataset.sel_unlabeled_ind, PL_dataset.target2, PL_dataset.sel_openset_ind, PL_dataset.target3, train_classes)
    NL_loader = DataLoader(NL_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    NL_optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    
    query_pool_ind = []
    query_pool_ind.extend(unlabeled_ind)
    query_pool_ind.extend(openset_ind)
    
    if known_openset_ind != []:
        query_pool_ind = [x for x in query_pool_ind if x not in pred_openset]
        
    query_loader = DataLoader(datasets['train'], batch_size=100, sampler=SubsetRandomSampler(query_pool_ind))
    query_prob = []
    query_idx = []
    
    unc_net = ResNet18(len(train_classes))
    unc_net = unc_net.to(device)
    checkpoint = torch.load('./checkpoint/ckpt_' + str(times) +'-best.pth')
    unc_net.load_state_dict(checkpoint['net'])
    idx_list, P_prob_list = test_PL(unc_net, query_loader, times, 0, device)
    
    count = 0
    break_count = 0
    select_num = 0
    if True:
        for epoch in range(args.nl_epochs + 1):
            if epoch % 5 ==0:
                count += 1
                idx_list, prob_list = test_NL(net, query_loader, times, epoch, device)
                if query_idx == []:
                    query_idx = idx_list
                    query_prob = prob_list
                else:
                    for i in range(len(prob_list)):
                        for j in range(len(train_classes)):
                            query_prob[i][j] = query_prob[i][j] + prob_list[i][j]

                if args.type == 1:
                    #LC
                    temp1 = np.asarray(query_prob).max(1)
                    temp1 = (temp1 - temp1.min())/(temp1.max()-temp1.min())
                    temp2 = np.asarray(P_prob_list).max(1)
                    temp2 = (temp2 - temp2.min())/(temp2.max()-temp2.min())
                elif args.type == 2:
                    #margin
                    prob = np.asarray(query_prob)
                    One_Two = np.argpartition(prob, -2, axis=1)[:,-2:]
                    margin = []
                    for i in range(One_Two.shape[0]):
                        margin.append(abs(prob[i,One_Two[i,1]] - prob[i,One_Two[i,0]]))
                    temp1 = np.asarray(margin)
                    temp1 = (temp1 - temp1.min())/(temp1.max()-temp1.min())
                    
                    
                    prob = np.asarray(P_prob_list)
                    One_Two = np.argpartition(prob, -2, axis=1)[:,-2:]
                    margin = []
                    for i in range(One_Two.shape[0]):
                        margin.append(abs(prob[i,One_Two[i,1]] - prob[i,One_Two[i,0]]))
                    temp2 = np.asarray(margin)
                    temp2 = (temp2 - temp2.min())/(temp2.max()-temp2.min())
                
                elif args.type == 3:
                    #Entropy
                    prob = np.asarray(query_prob)
                    predictions = np.sum(prob * np.log(prob + 1e-10), axis=1)
                    temp1 = np.asarray(predictions)
                    temp1 = (temp1 - temp1.min())/(temp1.max()-temp1.min())
                    
                    prob = np.asarray(P_prob_list)
                    predictions = np.sum(prob * np.log(prob + 1e-10), axis=1)
                    temp2 = np.asarray(predictions)
                    temp2 = (temp2 - temp2.min())/(temp2.max()-temp2.min())

                
                #Aux
                if known_openset_ind != []:
                    temp3 = np.asarray(Aux_prob2)
                    temp3 = (temp3 - temp3.min())/(temp3.max()-temp3.min())
                    temp = temp3*temp1 + ratio*(1-temp3)*temp2
                else:
                    temp = temp1 + ratio*temp2
                print(temp.shape)
                sel_pseudo_ind = np.asarray(query_idx)[np.argsort(temp)][:1500]
                    
                remove_from_unlabeled_ind = [x for x in sel_pseudo_ind if x in unlabeled_ind]
                remove_from_openset_ind = [x for x in sel_pseudo_ind if x in openset_ind]

                print("Select pseudo-ind accuracy: " + str(len(remove_from_unlabeled_ind)) +" / "+  str(len(sel_pseudo_ind)))
                logger.info("Select pseudo-ind accuracy: " + str(len(remove_from_unlabeled_ind)) +" / "+ str(len(sel_pseudo_ind)))

                if select_num == len(remove_from_unlabeled_ind):
                    break_count += 1
                    if break_count >= 5:
                        break
                else:
                    select_num = len(remove_from_unlabeled_ind)
                    break_count = 0



            NL_train(net, NL_loader, NL_optimizer, criterion_nll, len(train_classes), times, epoch, logger, device)

        #Aux
        sel_pseudo_ind = np.asarray(query_idx)[np.argsort(temp)][:1500]
        with open("sel_pseudo_ind_" + str(times) +".pkl", 'wb') as f:
            pickle.dump(sel_pseudo_ind, f)
        with open("prob_" + str(times) +".pkl", 'wb') as f:
            pickle.dump([query_idx, query_prob], f)    
    else:
        sel_pseudo_ind = random.sample(idx_list, 1500)
        
    remove_from_unlabeled_ind = [x for x in sel_pseudo_ind if x in unlabeled_ind]
    remove_from_openset_ind = [x for x in sel_pseudo_ind if x in openset_ind]
    
    ratio = len(remove_from_unlabeled_ind)/len(sel_pseudo_ind)
    
    openset_ind = [x for x in openset_ind if x not in remove_from_openset_ind]
    unlabeled_ind = [x for x in unlabeled_ind if x not in remove_from_unlabeled_ind]
    labeled_ind.extend(remove_from_unlabeled_ind)
    known_openset_ind.extend(remove_from_openset_ind)
    
    print("Select pseudo-ind accuracy: " + str(len(remove_from_unlabeled_ind)) +" / "+  str(len(sel_pseudo_ind)))
    logger.info("Select pseudo-ind accuracy: " + str(len(remove_from_unlabeled_ind)) +" / "+ str(len(sel_pseudo_ind)))
    