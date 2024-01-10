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


class SCFace(Dataset):

    def __init__(self, img_path):
        self._img_path = img_path
        self._samples = self._make_dataset()
        self.transform = transforms.Compose(
            [
             # transforms.Resize((112, 112)),
             transforms.Resize(112),
             transforms.CenterCrop(112),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.transform_flip = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(1),
             # transforms.Resize((112, 112)),
             transforms.Resize(112),
             transforms.CenterCrop(112),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        # print(f'found {len(self._samples)} images')

    def _make_dataset(self):
        
        img_list = os.listdir(self._img_path)
        new_list = []
        for img in img_list:
            new_list.append((img, int(img[:3])))
        return new_list
   
    def __getitem__(self, index):
        img_name, label = self._samples[index]
        img = PIL.Image.open(os.path.join(self._img_path, img_name))
        
        return self.transform(img), self.transform_flip(img)
     
    def __len__(self):
        return len(self._samples)
    
def visualize_feat(gallery_id, gallery_feat,
                   probe_id, probe_feat, 
                   reduct='tsne', num_id=20, title='matching in HR'):
    gallery_id = gallery_id.squeeze()
    probe_id = probe_id.squeeze()
    
    all_feat = np.zeros((6*num_id, gallery_feat.shape[1]))
    accu_index = 0
    for accu_id in range(num_id):
        id = gallery_id[accu_id]
        arr_indices_in_probe = np.where(probe_id == id)[0]
        all_feat[accu_index] = gallery_feat[accu_id]
        accu_index += 1
        for arr_index_in_probe in arr_indices_in_probe:
            all_feat[accu_index] = probe_feat[arr_index_in_probe]
            accu_index += 1
    
    if reduct == 'pca':
        all_feat_reducted = PCA(n_components=2).fit_transform(all_feat)
    elif reduct == 'tsne':
        all_feat_reducted = TSNE(n_components=2).fit_transform(all_feat)
    
    color_list = []
    edge_color_list = []
    for accu_id in range(num_id):
        color_list.extend([accu_id]*6)
        edge_color_list.extend(['black'])
        edge_color_list.extend(['none']*5)
    color_list = np.array(color_list)
    edge_color_list = np.array(edge_color_list)
    
    # probe first
    probe_index = [i for i in range(all_feat_reducted.shape[0]) if i%6!=0]
    plt.scatter(all_feat_reducted[probe_index, 0], all_feat_reducted[probe_index, 1], 
                c=color_list[probe_index], 
                cmap='rainbow',
                edgecolor=edge_color_list[probe_index])
    # gallery
    gallery_index = [i for i in range(all_feat_reducted.shape[0]) if i%6==0]
    plt.scatter(all_feat_reducted[gallery_index, 0], all_feat_reducted[gallery_index, 1], 
                c=color_list[gallery_index], 
                cmap='rainbow',
                edgecolor=edge_color_list[gallery_index])

    plt.xticks([]), plt.yticks([])
    plt.title(title)
    
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
    

    feature_list = []
    for loader in loader_list:
        with torch.no_grad():
            for i, (img, img_f) in enumerate(loader):
                # print(f'{i} / {len(loader)}')
                img = img.cuda()
                feat, norm = Resnet(img)
                # feat = feat * norm
                # feat = feat.cpu().numpy()
                
                img_f = img_f.cuda()
                feat_f, norm_f = Resnet(img_f)
                # feat_f = feat_f * norm_f
                # feat_f = feat_f.cpu().numpy()
                
                # feat = np.concatenate((feat, feat_f), axis=1)
                fused = feat + feat_f
                fused, _ = l2_norm(fused, axis=1)
                feat = fused.cpu().numpy()
                
                try:
                    features
                except NameError:
                    features = feat
                else:
                    features = np.concatenate((features, feat), axis=0)
                        
        feature_list.append(features)
        del features
    return feature_list



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
    parser.add_argument('--network', type=str, default='r50')
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
    dataset_root = "/mnt/HDD1/chiawei/datasets/SCface_landmarkCrop_match_Resolution"
    
    # gallery for d1 probe
    
    # gallery_for_probe_d1 = SCFace(img_path=os.path.join(dataset_root, 'gallery_for_p0'))
    gallery_for_probe_d1 = SCFace(img_path=os.path.join(dataset_root, 'gallery'))

    gallery_for_probe_d1_db = DataLoader(dataset=gallery_for_probe_d1,
                                    batch_size=256,
                                    num_workers=6,
                                    pin_memory=True,
                                    shuffle=False)
    gallery_for_probe_d1_id = extract_id_from_tuple(gallery_for_probe_d1._samples)
    
    # gallery for d2 probe
    
    # gallery_for_probe_d2 = SCFace(img_path=os.path.join(dataset_root, 'gallery_for_p1'))
    gallery_for_probe_d2 = SCFace(img_path=os.path.join(dataset_root, 'gallery'))
    gallery_for_probe_d2_db = DataLoader(dataset=gallery_for_probe_d2,
                                    batch_size=256,
                                    num_workers=6,
                                    pin_memory=True,
                                    shuffle=False)
    gallery_for_probe_d2_id = extract_id_from_tuple(gallery_for_probe_d2._samples)
    
    # gallery for d3 probe
    
    # gallery_for_probe_d3 = SCFace(img_path=os.path.join(dataset_root, 'gallery_for_p2'))
    gallery_for_probe_d3 = SCFace(img_path=os.path.join(dataset_root, 'gallery'))
    gallery_for_probe_d3_db = DataLoader(dataset=gallery_for_probe_d3,
                                    batch_size=256,
                                    num_workers=6,
                                    pin_memory=True,
                                    shuffle=False)
    gallery_for_probe_d3_id = extract_id_from_tuple(gallery_for_probe_d3._samples)
    
    # probe_d1
    probe_d1 = SCFace(img_path=os.path.join(dataset_root, 'probe_d1'))

    probe_d1_db = DataLoader(dataset=probe_d1,
                                    batch_size=256,
                                    num_workers=6,
                                    pin_memory=True,
                                    shuffle=False)
    probe_d1_id = extract_id_from_tuple(probe_d1._samples)
    
    # probe_d2
    probe_d2 = SCFace(img_path=os.path.join(dataset_root, 'probe_d2'))

    probe_d2_db = DataLoader(dataset=probe_d2,
                                    batch_size=256,
                                    num_workers=6,
                                    pin_memory=True,
                                    shuffle=False)
    probe_d2_id = extract_id_from_tuple(probe_d2._samples)
    
    # probe_d3
    probe_d3 = SCFace(img_path=os.path.join(dataset_root, 'probe_d3'))

    probe_d3_db = DataLoader(dataset=probe_d3,
                                    batch_size=256,
                                    num_workers=6,
                                    pin_memory=True,
                                    shuffle=False)
    probe_d3_id = extract_id_from_tuple(probe_d3._samples)
    
  
    loader_list = [gallery_for_probe_d1_db, gallery_for_probe_d2_db, gallery_for_probe_d3_db, 
                   probe_d1_db, probe_d2_db, probe_d3_db]
    """ 
    ######################################
    ###  Ours   forward  #################
    ######################################
    """
    # ckp_path_list = ["/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e0_iter35406_loss_arcface2.78_loss_lip0.000000.pth", # 63.38, 98.31, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e1_iter35406_loss_arcface2.21_loss_lip0.000000.pth", # 56.77, 97.08, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e2_iter35406_loss_arcface2.28_loss_lip0.000000.pth", # 66.46, 98.46, 99.85
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e3_iter35406_loss_arcface2.00_loss_lip0.000000.pth", # 65.23, 98, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e4_iter35406_loss_arcface1.71_loss_lip0.000000.pth"] # 64.92, 97.85, 99.85
    
    
    # ckp_path_list = ["/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e0_iter35406_loss_arcface2.79_loss_lip0.001038.pth", # 70, 98.92, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e1_iter35406_loss_arcface2.26_loss_lip0.000892.pth", # 68, 98.92, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e2_iter35406_loss_arcface2.33_loss_lip0.001022.pth", # 70.61, 99.07, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e3_iter35406_loss_arcface2.16_loss_lip0.000775.pth", # 69.53, 98.77, 99.85
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e4_iter35406_loss_arcface1.82_loss_lip0.001197.pth"] # 69.08, 98.92, 99.85
    # ckp_path_list = ["/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e0_iter35406_loss_arcface2.91_loss_lip0.000698.pth", # 70.92, 99.23, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e1_iter35406_loss_arcface2.42_loss_lip0.000384.pth", # 69.84, 99.08, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e2_iter35406_loss_arcface2.44_loss_lip0.000826.pth", # 72, 99.38, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e3_iter35406_loss_arcface2.37_loss_lip0.000619.pth", # 70.46, 99.23, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e4_iter35406_loss_arcface1.98_loss_lip0.000468.pth"] # 69.23, 99.23, 100
    # ckp_path_list = ["/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e0_iter35406_loss_arcface2.80_loss_lip0.001211.pth", # 72.77, 99.23, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e1_iter35406_loss_arcface2.25_loss_lip0.000730.pth", # 70.92, 99.23, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e2_iter35406_loss_arcface2.36_loss_lip0.001329.pth", # 72.15, 99.35, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e3_iter35406_loss_arcface2.16_loss_lip0.000850.pth", # 71.85, 99.23, 100
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e4_iter35406_loss_arcface1.84_loss_lip0.000794.pth"] # 72.15, 99.23, 100
    ckp_path_list = [
                    "/mnt/HDD1/yuwei/insightface_lipface/work_dirs/ms1mv2_r50_lip_ngpu2_p05_lrate001/model_e4.pt",
                    "/mnt/HDD1/yuwei/insightface_lipface/work_dirs/ms1mv2_r50_arc_ngpu2/model_e19.pt"
    ]
    # ckp_path_list = os.listdir(root)
    # for i, ckp_path in enumerate(ckp_path_list):
    #     ckp_path_list[i] = os.path.join(root, ckp_path)
    # ckp_path_list.sort(key=lambda x: os.path.getmtime(x))
    # ckp_path_list.sort()
    from collections import OrderedDict
    def clean_dict_inf(model, state_dict):
        _state_dict = OrderedDict()
        for k, v in state_dict.items():
            # # assert k[0:1] == 'features.module.'
            new_k = 'features.'+'.'.join(k.split('.')[2:])
            #print(k, new_k)
            if new_k in model.state_dict().keys() and \
            v.size() == model.state_dict()[new_k].size():
                _state_dict[new_k] = v
            # assert k[0:1] == 'module.features.'
            new_kk = '.'.join(k.split('.')[2:])
            #print(new_kk)
            if new_kk in model.state_dict().keys() and \
            v.size() == model.state_dict()[new_kk].size():
                _state_dict[new_kk] = v
        num_model = len(model.state_dict().keys())
        num_ckpt = len(_state_dict.keys())
        if num_model != num_ckpt:
            sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
                num_model, num_ckpt))
        return _state_dict

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
        
        gallery_for_probe_d1_feat, gallery_for_probe_d2_feat, gallery_for_probe_d3_feat, \
            probe_d1_feat, probe_d2_feat, probe_d3_feat = resnet_forward(Resnet, loader_list)
        
        # subtract mean
        gallery_for_probe_d1_feat -= np.mean(gallery_for_probe_d1_feat, axis=0)
        gallery_for_probe_d2_feat -= np.mean(gallery_for_probe_d2_feat, axis=0)
        gallery_for_probe_d3_feat -= np.mean(gallery_for_probe_d3_feat, axis=0)
        probe_d1_feat -= np.mean(probe_d1_feat, axis=0)
        probe_d2_feat -= np.mean(probe_d2_feat, axis=0)
        probe_d3_feat -= np.mean(probe_d3_feat, axis=0)
        
        # normalize
        gallery_for_probe_d1_feat = normalize(gallery_for_probe_d1_feat)
        gallery_for_probe_d2_feat = normalize(gallery_for_probe_d2_feat)
        gallery_for_probe_d3_feat = normalize(gallery_for_probe_d3_feat)
        probe_d1_feat = normalize(probe_d1_feat)
        probe_d2_feat = normalize(probe_d2_feat)
        probe_d3_feat = normalize(probe_d3_feat)
        
        for gallery_feat, probe_feat, gallery_id, probe_id in [(gallery_for_probe_d1_feat, probe_d1_feat, gallery_for_probe_d1_id, probe_d1_id), 
                                                                (gallery_for_probe_d2_feat, probe_d2_feat, gallery_for_probe_d2_id, probe_d2_id),
                                                                (gallery_for_probe_d3_feat, probe_d3_feat, gallery_for_probe_d3_id, probe_d3_id)]:

        
            
            #print("#"*20 + " calculating acc!  " + "#"*20)
            
            mAP, CMC = calculate_acc(gallery_id,
                                     probe_id,
                                     gallery_feat,
                                     probe_feat)
            # print(f'mAP = {mAP}')
            print(f'r1 precision = {CMC[0]}')
            #print(f'r5 precision = {CMC[4]}')
            # print(f'r10 precision = {CMC[9]}')
            #print(f'r20 precision = {CMC[19]}')
        
    # visualize_feat(gallery_for_probe_d1_id, gallery_for_probe_d1_feat,
    #                probe_d1_id, probe_d1_feat, 
    #                reduct='tsne', num_id=60, title='matching in LR')
    
    # false_list = find_false_case(gallery_for_probe_d1._samples, 
    #                              probe_d1._samples, 
    #                              gallery_for_probe_d1_feat, 
    #                              probe_d1_feat, 
    #                              os.path.join(dataset_root, 'gallery_for_p0'),
    #                              os.path.join(dataset_root, 'probe_d1'),)