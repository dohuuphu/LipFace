# This file extract the features of test images.
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from scipy.io import loadmat, savemat
import torch
from torch import nn
torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#import torchvision.transforms
import PIL.Image
from PIL import Image
from sklearn.preprocessing import normalize
import numpy as np

import warnings
warnings.simplefilter("ignore")

# from joblib import Parallel, delayed
# import multiprocessing

from backbones import get_model
from backbones_curricularface.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import backbone_adaface.net
#from myUtil import filter_module
#import PIL.Image
import matplotlib.pyplot as plt
from torchvision import transforms
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# import argparse
# import numpy as np
# import pandas as pd
# from os import makedirs
# from os.path import dirname, join
# from glob import glob
# from termcolor import cprint
from tqdm import tqdm
# from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import paired_distances
from io import BytesIO
#from utils import filter_module

import argparse

class LFW(Dataset):

    def __init__(self, img_path, pair_txt_path, LR_resolution, match_at_LR):
        self._img_path = img_path
        self._pair_txt_path = pair_txt_path
        self._LR_resolution = LR_resolution
        self._match_at_LR = match_at_LR
        self._samples = self._make_dataset()
        self.transform = transforms.Compose(
            [
             transforms.Resize(112),
             transforms.CenterCrop(112),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.transform_flip = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(1),
             transforms.Resize(112),
             transforms.CenterCrop(112),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        # print(f'found {len(self._samples)} images')
        

    def _make_dataset(self):
        samples = []

        f = open(self._pair_txt_path)
        for l, line in enumerate(f.readlines()):
            # print(l)
            if l == 0:
                num_folds, match_num_each_fold = line.split() # 10, 300 (match_num_each_fold = mismatch_num_each_fold)
                match_num_each_fold = int(match_num_each_fold)
            else:
                # fold 0: l = 1~300 (match), 301~600 (mismatch)
                # fold 1: l = 601~900 (match), 901~1200 (mismatch)
                # ...
                # fold 9: l = 5401~5700 (match), 5701~6000 (mismatch)
                now_fold = (l-1) // (match_num_each_fold*2)
                if (l-1) < now_fold * (match_num_each_fold*2) + match_num_each_fold:
                    # l = 300, 299<0*600+300=300 (match)
                    # l = 301, 300!<0*600+300=300 (mismatch)
                    # l = 601, 600<1*600+300=900 (match)
                    # l = 900, 899<1*600+300=900 (match)
                    # l = 901, 900!<1*600+300=900 (mismatch)
                    # match
                    img_id, img1_name, img2_name = line.split()
                    img1_path = os.path.join(self._img_path, img_id, img_id+'_'+img1_name.zfill(4)+'.png')
                    img2_path = os.path.join(self._img_path, img_id, img_id+'_'+img2_name.zfill(4)+'.png')
                    # img1_path = os.path.join(self._img_path, img_id, img_id+'_'+img1_name.zfill(4)+'.jpg')
                    # img2_path = os.path.join(self._img_path, img_id, img_id+'_'+img2_name.zfill(4)+'.jpg')
                else:
                    # mismatch
                    img1_id, img1_name, img2_id, img2_name = line.split()
                    img1_path = os.path.join(self._img_path, img1_id, img1_id+'_'+img1_name.zfill(4)+'.png')
                    img2_path = os.path.join(self._img_path, img2_id, img2_id+'_'+img2_name.zfill(4)+'.png')
                    # img1_path = os.path.join(self._img_path, img1_id, img1_id+'_'+img1_name.zfill(4)+'.jpg')
                    # img2_path = os.path.join(self._img_path, img2_id, img2_id+'_'+img2_name.zfill(4)+'.jpg')
                
                samples.append((img1_path, 'HR'))
                samples.append((img2_path, 'LR'))
            
        f.close()
        return samples
        
   
    def __getitem__(self, index):
        img_path, HR_or_LR = self._samples[index]
        img = PIL.Image.open(img_path)
        
        # if HR_or_LR == 'LR':
        #     buffer = BytesIO()
        #     img.resize((self._LR_resolution, self._LR_resolution), Image.NEAREST).save(buffer, "JPEG", quality=65)
        #     img = PIL.Image.open(buffer)
        #     # img = img.resize((self._LR_resolution, self._LR_resolution), Image.NEAREST)
        #     # buffer.close()
        # elif HR_or_LR == 'HR':
        #     if self._match_at_LR:
        #         # img = img.resize((self._LR_resolution, self._LR_resolution), Image.NEAREST)
        #         buffer = BytesIO()
        #         img.resize((self._LR_resolution, self._LR_resolution), Image.NEAREST).save(buffer, "JPEG", quality=65)
        #         img = PIL.Image.open(buffer)
        
        return self.transform(img), self.transform_flip(img)
     
    def __len__(self):
        return len(self._samples)
    
def test_buffer():

    img = PIL.Image.open("/DISK2/chiawei/LFW/lfw_correct_aligned/Aaron_Eckhart/Aaron_Eckhart_0001.png")
    plt.imshow(img)
    
    plt.figure()
    buffer = BytesIO()
    
    img.resize((112,112), Image.NEAREST).save(buffer, "JPEG", quality=1)
    new_img = PIL.Image.open(buffer)
    plt.imshow(new_img)

    
"""
################################################
### Resnet Testing  ############################
################################################
"""
def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm

def resnet_forward(Resnet, loader):

    with torch.no_grad():
        for i, (img, img_f) in enumerate(loader):
            # print(f'{i} / {len(loader)}')
            img = img.cuda()
            feat, _, _ = Resnet(img)
            feat = feat.cpu().numpy()
            
            # including flip
            img_f = img_f.cuda()
            feat_f, _, _ = Resnet(img_f)
            feat_f = feat_f.cpu().numpy()
            
            feat = np.concatenate((feat, feat_f), axis=1)
            # fused = feat + feat_f
            # fused, _ = l2_norm(fused, axis=1)
            # feat = fused.cpu().numpy()
            
            try:
                features
            except NameError:
                features = feat
            else:
                features = np.concatenate((features, feat), axis=0)

    return features

def k_fold_cross_validation(embs: np.array, labels: np.array, metric: str = "cosine", k: int = 10):
    # https://github.com/Martlgap/xqlfw/blob/main/code/evaluate.py
    """ Perform k-fold cross validation
    :param embs: embeddings (feature vectors) of pairs in consecutively a list [emb0a, emb0b, emb_1a, emb_1b, ...]
    :param labels: list with booleans for embeddings pairs [True, False, ...] len(labels) = len(embeddings) / 2
    :param metric: which metric to use for distance between embeddings, i.e. 'cosine' or 'euclidean', ...
    :param k: number of folds to use
    :return: for each fold -> true positive rates, false positive rates, accuracies, thresholds
    """

    def _evaluate(_thresh: float, _dists: np.array, _labels: np.array):
        """ Evaluate TP, FP, TN, and FN -> calculate accuracy
        :param _thresh:
        :param _dists: array of distances
        :param _labels:
        :return:
        """
        predictions = np.less(_dists, _thresh)
        tp = np.sum(np.logical_and(predictions, _labels))
        fp = np.sum(np.logical_and(predictions, np.logical_not(_labels)))
        tn = np.sum(np.logical_and(np.logical_not(predictions), np.logical_not(_labels)))
        fn = np.sum(np.logical_and(np.logical_not(predictions), _labels))
        actual_tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        actual_fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / _dists.size
        return actual_tpr, actual_fpr, acc

    embs_1 = embs[0::2]
    embs_2 = embs[1::2]

    nrof_pairs = len(labels)
    thresholds = np.arange(0, 2, 0.001)
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=k, shuffle=False)

    tprs = np.zeros((k, nrof_thresholds))
    fprs = np.zeros((k, nrof_thresholds))
    accuracies = np.zeros(k)
    best_thresholds = np.zeros(k)
    indices = np.arange(nrof_pairs)

    dists = paired_distances(embs_1, embs_2, metric=metric)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # train_set = np.array(train_set).astype(int)
        # print(train_set)
        # Find the best threshold for the fold
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold_train in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = _evaluate(threshold_train, dists[train_set], np.array(labels)[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold_test in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = _evaluate(
                threshold_test, dists[test_set], np.array(labels)[test_set]
            )

        _, _, accuracies[fold_idx] = _evaluate(
            thresholds[best_threshold_index], dists[test_set], np.array(labels)[test_set]
        )

    return tprs, fprs, accuracies, best_thresholds



if __name__ == '__main__':

   
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='r50')
    args = parser.parse_args()
    Resnet = get_model(args.network, dropout=0, fp16=False).cuda()
   
    ckp_path_list = [
                    # "/mnt/HDD1/phudh/ICIP/insightface_lipface/work_dirs/ms1mv2_r50_lip_ngpu2_p05_lrate001_lr42_MixDegraded_attn_8epoch/model_e7.pt",
                    # "/mnt/HDD1/phudh/ICIP/insightface_lipface/work_dirs/ms1mv2_r50_lip_ngpu2_p05_lrate001_lr42_originalDegraded_attn/model_e5.pt"
                    "/mnt/HDD1/phudh/ICIP/insightface_lipface/work_dirs/ms1mv2_r50_lip_ngpu2_p05_lrate001_lr42_MixDegraded_attn_8epoch_resume/model_e7.pt",
                    "/mnt/HDD1/phudh/ICIP/insightface_lipface/work_dirs/ms1mv2_r50_lip_ngpu2_p05_lrate001_lr42_MixDegraded_attn_8epoch_resume/model_e6.pt"
                     ]
    
    for ckp_path in ckp_path_list:
        print(ckp_path)
        Resnet.load_state_dict(torch.load(ckp_path))
        #Resnet = nn.DataParallel(Resnet)
        # statedict = torch.load(ckp_path)['state_dict']
        # model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        # Resnet.load_state_dict(model_statedict)
        Resnet.eval()
        """ datasets """
        
        
        
        # img_path = "/DISK2/chiawei/LFW/lfw_correct_aligned_random_margin_padding_0.6/"

        # lfw
        img_path = "/mnt/HDD1/phudh/ICIP/insightface_lipface/data_test/LFW/lfw_correct_aligned"
        pair_txt_path = "/mnt/HDD1/phudh/ICIP/insightface_lipface/data_test/LFW/LFW_pairs.txt"

        #xqlfw
        # img_path = "/mnt/HDD1/phudh/ICIP/insightface_lipface/data_test/XQLFW/xqlfw_aligned_112"
        # pair_txt_path = "/mnt/HDD1/phudh/ICIP/insightface_lipface/data_test/XQLFW/xqlfw_pairs.txt"

        lfw = LFW(img_path, pair_txt_path, LR_resolution=112, match_at_LR=False)

        loader = DataLoader(dataset=lfw,
                            batch_size=256,
                            num_workers=6,
                            pin_memory=True,
                            shuffle=False)
        
        """ 
        ######################################
        ###  Ours   forward  #################
        ######################################
        """
        
        # print("#"*20 + " forwarding thru Ours!  " + "#"*20)
        
        features = resnet_forward(Resnet, loader)
        
        # correlation distance
        features -= np.mean(features, axis=0)
        # normalize
        features = normalize(features)
        
        """ start eval """
        # to list of numpy features
        feature_list = []
        for f in features:
            feature_list.append(f)
        # there are totally 6000 pairs = 10folds*(300match+300mismatch)
        # hence 6000*2 num of features
        assert len(feature_list) == 6000*2
        
        match_gt = []
        for n_folds in range(10):
            for n_match in range(300):
                match_gt.append(True)
            for n_mismatch in range(300):
                match_gt.append(False)
        assert len(match_gt) == 6000
        
        tprs, fprs, accuracies, best_thresholds = k_fold_cross_validation(feature_list, match_gt)
        print(np.mean(accuracies))