# This file extract the features of test images.
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'

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
from sklearn.preprocessing import normalize
import numpy as np

import warnings
warnings.simplefilter("ignore")

from joblib import Parallel, delayed
import multiprocessing

from backbones import get_model
from backbones_curricularface.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import backbone_adaface.net
import backbone_magface.iresnet
#from utils import filter_module
#import PIL.Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import argparse
from dataloader_vggface2 import Vggface2
    
def visualize_feat(label_hr, feat_hr,
                   label_lr, feat_lr, 
                   reduct='tsne', num_id=3, title=''):
    feat_hr = np.array(feat_hr)
    feat_hr = feat_hr.reshape((722, 512))
    feat_lr = np.array(feat_lr)
    feat_lr = feat_lr.reshape((722, 512))
    combined_features = np.concatenate((feat_hr, feat_lr), axis=0)
    label_hr = np.array(label_hr)
    label_lr = np.array(label_lr)
    combined_labels = np.concatenate((label_hr, label_lr), axis=1)
    print(combined_labels.shape)

    tsne = TSNE(n_components=2, random_state=0)

    tsne_result = tsne.fit_transform(combined_features)

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 30})
    plt.scatter(tsne_result[:722, 0], tsne_result[:722, 1], c=label_hr.ravel(), cmap=plt.cm.tab20b, s=100, marker='s')
    plt.scatter(tsne_result[722:, 0], tsne_result[722:, 1], c=label_lr.ravel(), cmap=plt.cm.tab20b, s=10, marker='^')
    plt.xticks([]), plt.yticks([])
    plt.legend()
    plt.savefig("tsne_lip_surv_ms1mv3.png")
    
def compute_AP(good_image, index):
    cmc = np.zeros((len(index), 1))
    ngood = len(good_image)

    old_recall = 0.
    old_precision = 1.
    ap = 0.
    intersect_size = 0.
    j = 0.
    good_now = 0.

    for n in range(len(index)):
        flag = 0
        if index[n] in good_image:
            cmc[n:] = 1
            flag = 1
            good_now = good_now + 1
        if flag == 1:
            intersect_size += 1
        recall = intersect_size / ngood
        precision = intersect_size / (j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        j += 1

        if good_now == ngood:
            break

    return ap, cmc.T


def calculate_acc(gallery_id, probe_id, gallery_feature, probe_feature):
    
    assert gallery_feature.shape == (130, 512)
    assert probe_feature.shape == (130*5, 512)

    assert gallery_id.shape[0] == 130
    assert probe_id.shape[0] == 130*5

    # L2 distance
    dist = torch.cdist(torch.from_numpy(gallery_feature), \
                        torch.from_numpy(probe_feature)).numpy() # (157871,3728) = (#gallery, #probe)
    assert dist.shape == (gallery_feature.shape[0], probe_feature.shape[0])
    # cosine distance
    # dist = torch.matmul(torch.from_numpy(gallery_feature_map), torch.from_numpy(probe_feature_map).T).numpy()
        
    _CMC = np.zeros((probe_feature.shape[0], gallery_feature.shape[0])) # (3728,157871)
    ap = np.zeros((probe_feature.shape[0]))
    
    num_cores = multiprocessing.cpu_count()
        
    # x = Parallel(n_jobs=num_cores)(delayed(compute_AP)(np.where(gallery_ids == probe_ids[p,0])[0], np.argsort(dist[:, p])[::-1]) for p in range(probe_feature_map.shape[0]))
    x = Parallel(n_jobs=num_cores)(delayed(compute_AP)(np.where(gallery_id == probe_id[p,0])[0], np.argsort(dist[:, p])) for p in range(probe_feature.shape[0]))
    for i, (_ap, _cmc) in enumerate(x):
        ap[i] = _ap
        _CMC[i, :] = _cmc
        
    CMC = np.mean(_CMC, axis=0)  

    return np.mean(ap), CMC

    
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

def resnet_forward(Resnet, loader_list):
    label_list = []
    feature_hr_list = []
    feature_lr_list = []
    for loader in loader_list:
        with torch.no_grad():
            for i, (img, label) in enumerate(loader):
                print(img.shape, label.shape)
                # print(f'{i} / {len(loader)}')
                # reshape
                img = torch.reshape(img, (len(label)*(1+1), 3, 112, 112))
                img = img.cuda()
                feat, norm = Resnet(img)
                feat = torch.div(feat, norm)
                feat = torch.reshape(feat, (len(label), (1+1), 512))
                assert not torch.isnan(feat).any()

                try:
                    features_hr
                except NameError:
                    features_hr = feat[:, 0, :].cpu().numpy()
                else:
                    features_hr = np.concatenate((features_hr, feat[:, 0, :].cpu().numpy()), axis=0)
                
                try:
                    features_lr
                except NameError:
                    features_lr = feat[:, 1, :].cpu().numpy()
                else:
                    features_lr = np.concatenate((features_lr, feat[:, 1, :].cpu().numpy()), axis=0)

                try:
                    labels
                except NameError:
                    labels = label.cpu().numpy()
                else:
                    labels = np.concatenate((labels, label.cpu().numpy()), axis=0)
                
        label_list.append(labels)
        feature_hr_list.append(features_hr)
        feature_lr_list.append(features_lr)
        del labels, features_hr, features_lr
                
                
    return label_list, feature_hr_list, feature_lr_list



def extract_id_from_tuple(list_of_tuple):
    id_list = []
    for tuple in list_of_tuple:
        id_list.append(tuple[1])
    return np.array(id_list)[:,np.newaxis]

def find_false_case(gallery_id_name, 
                    probe_id_name, 
                    gallery_feature, 
                    probe_feature, 
                    gallery_img_root,
                    probe_img_root):
    
    probe_id = extract_id_from_tuple(probe_id_name)
    gallery_id = extract_id_from_tuple(gallery_id_name)
    
    assert gallery_feature.shape == (130, 512*2)
    assert probe_feature.shape == (130*5, 512*2)

    assert gallery_id.shape[0] == 130
    assert probe_id.shape[0] == 130*5

    # L2 distance
    dist = torch.cdist(torch.from_numpy(gallery_feature), \
                        torch.from_numpy(probe_feature)).numpy() # (157871,3728) = (#gallery, #probe)
    assert dist.shape == (gallery_feature.shape[0], probe_feature.shape[0])
    
    ret = []
    
    for p in range(probe_feature.shape[0]):
        # probe one by one, find the most similar gallery
        gallery_index_sim = np.argsort(dist[:, p])[0]
        gallery_index_gt = np.where(gallery_id == probe_id[p,0])[0]
        if gallery_index_sim not in gallery_index_gt:
            # a false case
            probe_img_path = probe_id_name[p][0]
            gt_gallery_img_path = gallery_id_name[gallery_index_gt[0]][0]
            pred_gallery_img_path = gallery_id_name[gallery_index_sim][0]
            
            ret.append((probe_img_path, gt_gallery_img_path, pred_gallery_img_path))
            
            plt.figure()
            
            plt.subplot(131)
            img = PIL.Image.open(os.path.join(probe_img_root, probe_img_path))
            plt.imshow(img)
            plt.axis('off')
            plt.title('probe')
            
            plt.subplot(132)
            img = PIL.Image.open(os.path.join(gallery_img_root, gt_gallery_img_path))
            plt.imshow(img)
            plt.axis('off')
            plt.title('gallery gt')
            
            plt.subplot(133)
            img = PIL.Image.open(os.path.join(gallery_img_root, pred_gallery_img_path))
            plt.imshow(img)
            plt.axis('off')
            plt.title('gallery pred')
            
    return ret



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='r100')
    args = parser.parse_args()
    # Resnet = backbone_magface.iresnet.iresnet100(
    #         pretrained=False,
    #         num_classes=512,
    #     ).cuda()
    Resnet = get_model(args.network, dropout=0, fp16=False).cuda()
    # Resnet = IR_101((112, 112)).cuda()
    # Resnet = backbone_adaface.net.build_model('ir_101').cuda()
  
    """ datasets """
    #dataset_root = '/DISK2/chiawei/SCface_landmarkCrop_match_Resolution/'
    dataset_root = "/mnt/HDD1/yuwei/dataset/vggface2_toy"
    
    # dataloader
    vggface2 = Vggface2(root=dataset_root)
    test_loader = DataLoader(dataset=vggface2,
                                 batch_size=128,
                                 num_workers=6,
                                 shuffle=True,
                                 pin_memory=True,
                                 drop_last=False) 

    test_len = len(test_loader)
    
  
    loader_list = [test_loader]
    """ 
    ######################################
    ###  Ours   forward  #################
    ######################################
    """
    ckp_path_list = [
                    # "/mnt/HDD1/yuwei/insightface_lipface/work_dirs/ms1mv3_r100_lip_finetune_p005/model_e0.pt"
                    # "/mnt/HDD1/yuwei/insightface_lipface/work_dirs/ms1mv3_r100_pre/backbone.pth"
                    "/mnt/HDD1/yuwei/insightface_lipface/work_dirs/ms1mv3_r100_lip_finetune_surviliance/model_e0.pt"
    ]

    for ckp_path in ckp_path_list:
        print(ckp_path)
        # if (os.path.basename(ckp_path)[0] != 'e'):
        #     continue
        # print(os.path.basename(ckp_path))
        
        #Resnet.load_state_dict(filter_module(torch.load(ckp_path)['Resnet']))
        Resnet.load_state_dict(torch.load(ckp_path))
        # checkpoint = torch.load(ckp_path)
        # _state_dict = clean_dict_inf(Resnet, checkpoint['state_dict'])
        # model_dict = Resnet.state_dict()
        # model_dict.update(_state_dict)
        # Resnet.load_state_dict(model_dict)
        # statedict = torch.load(ckp_path)['state_dict']
        # model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        # Resnet.load_state_dict(model_statedict)
        Resnet.eval()
    
        # print("#"*20 + " forwarding thru Ours!  " + "#"*20)
        
        label, feat_hr, feat_lr = resnet_forward(Resnet, loader_list)
        
        visualize_feat(label, feat_hr, label, feat_lr, reduct='tsne', num_id=3, title='')
    
    # false_list = find_false_case(gallery_for_probe_d1._samples, 
    #                              probe_d1._samples, 
    #                              gallery_for_probe_d1_feat, 
    #                              probe_d1_feat, 
    #                              os.path.join(dataset_root, 'gallery_for_p0'),
    #                              os.path.join(dataset_root, 'probe_d1'),)