from torch.utils.data import Dataset
import pickle
import torchvision.transforms as transforms
import numpy as np
import copy
import random
    

class PL_Dataset(Dataset): 
    def __init__(self, datasets, labeled_ind, unlabeled_ind, openset_ind, train_classes): 
        super(PL_Dataset, self).__init__()
        
        NL_batch = 10000
        
        ratio = len(unlabeled_ind)/(len(unlabeled_ind) + len(openset_ind))
        sel_unlabeled_ind = random.sample(unlabeled_ind, int(ratio*NL_batch))
        sel_openset_ind = random.sample(openset_ind, NL_batch - int(ratio*NL_batch))
        self.data1 = copy.deepcopy(datasets['train'].data)[labeled_ind].tolist()
        self.data2 = copy.deepcopy(datasets['train'].data)[sel_unlabeled_ind].tolist()
        self.data3 = copy.deepcopy(datasets['train'].data)[sel_openset_ind].tolist()
        
        self.target1 = [datasets['train'].targets[x] for x in labeled_ind]
        
        target_xform_dict = {}
        for i, k in enumerate(train_classes):
            target_xform_dict[k] = i
        for ind in range(len(self.target1)):  
            self.target1[ind] = target_xform_dict[self.target1[ind]]
        
        self.target2 = [-1 for i in range(len(sel_unlabeled_ind))]
        self.target3 = [-1 for i in range(len(sel_openset_ind))]
        
        self.data = []
        self.data.extend(self.data1)
        self.data.extend(self.data2)
        self.data.extend(self.data3)
        self.data = np.asarray(self.data)
        
        self.labels = []
        self.labels.extend(self.target1)
        self.labels.extend(self.target2)
        self.labels.extend(self.target3)
        
        self.ind = []
        self.ind.extend(labeled_ind)
        self.ind.extend(sel_unlabeled_ind)
        self.ind.extend(sel_openset_ind)
        
        
        self.ToPILImage = transforms.ToPILImage()
        self.transform = datasets['train'].transform
        self.labeled_ind = labeled_ind
        self.sel_unlabeled_ind = sel_unlabeled_ind
        self.sel_openset_ind = sel_openset_ind
        self.train_classes = train_classes
        
        
    def __len__(self): 
        return len(self.labels)
    
    def __getitem__(self, idx): 
        data = self.ToPILImage(np.uint8(self.data[idx]))
        if self.transform:
            data = self.transform(data)
        return data, self.labels[idx], self.ind[idx], idx
    
    def get_test_ind(self):
        test_ind = []
        for e in self.sel_unlabeled_ind:
            test_ind.append(self.ind.index(e))
        for e in self.sel_openset_ind:
            test_ind.append(self.ind.index(e))
        return test_ind
    
    def get_train_ind(self):
        train_ind = []
        for e in self.labeled_ind:
            train_ind.append(self.ind.index(e))
        return train_ind
    
class NL_Dataset(Dataset): 
    def __init__(self, datasets, labeled_ind, labeled_label, unlabeled_ind, unlabeled_label, openset_ind, openset_label, train_classes): 
        super(NL_Dataset, self).__init__()
        
        sel_unlabeled_ind = unlabeled_ind
        sel_openset_ind = openset_ind
        self.data1 = copy.deepcopy(datasets['train'].data)[labeled_ind].tolist()
        self.data2 = copy.deepcopy(datasets['train'].data)[sel_unlabeled_ind].tolist()
        self.data3 = copy.deepcopy(datasets['train'].data)[sel_openset_ind].tolist()
        
        self.target1 = labeled_label
        self.target2 = unlabeled_label
        self.target3 = openset_label
        
        self.data = []
        self.data.extend(self.data1)
        self.data.extend(self.data2)
        self.data.extend(self.data3)
        self.data = np.asarray(self.data)
        
        self.labels = []
        self.labels.extend(self.target1)
        self.labels.extend(self.target2)
        self.labels.extend(self.target3)
        
        self.ind = []
        self.ind.extend(labeled_ind)
        self.ind.extend(sel_unlabeled_ind)
        self.ind.extend(sel_openset_ind)
        
        
        self.ToPILImage = transforms.ToPILImage()
        self.transform = datasets['train'].transform
        self.labeled_ind = labeled_ind
        self.sel_unlabeled_ind = sel_unlabeled_ind
        self.sel_openset_ind = sel_openset_ind
        self.train_classes = train_classes
        
    def __len__(self): 
        return len(self.labels)
    
    def __getitem__(self, idx): 
        data = self.ToPILImage(np.uint8(self.data[idx]))
        if self.transform:
            data = self.transform(data)
        if self.labels[idx] == -1:
            label = random.randint(0, len(self.train_classes)-1)
            return data, label, self.ind[idx], idx
        return data, self.labels[idx], self.ind[idx], idx
    
    def get_test_ind(self):
        test_ind = []
        for e in self.sel_unlabeled_ind:
            test_ind.append(self.ind.index(e))
        for e in self.sel_openset_ind:
            test_ind.append(self.ind.index(e))
        return test_ind
    
    def get_train_ind(self):
        train_ind = []
        for e in self.labeled_ind:
            train_ind.append(self.ind.index(e))
        return train_ind    
