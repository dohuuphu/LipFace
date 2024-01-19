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
import torchvision.transforms
import PIL.Image
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore")

from joblib import Parallel, delayed
import multiprocessing

#from backbone import iresnet100, iresnet50, iresnet34
from backbones import get_model
from backbones_curricularface.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
import backbone_adaface.net
import backbone_magface.iresnet
from torchvision import transforms
#from utils import filter_module
import argparse


class Tinyface(Dataset):

    def __init__(self, img_path, mat_id_path=None):
        self._img_path = img_path
        self._dataset = self._find_dataset(img_path)
        self._samples = self._make_dataset(mat_id_path)
        print(f'found {len(self._samples)} images')
        self.transform = transforms.Compose(
            [
             # transforms.Resize((100, 100)),
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
#             transforms.Resize((112, 112)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        # print(f'found {len(self._samples)} images')
    
    def _find_dataset(self, img_path):
        last = os.path.basename(img_path)
        if last == 'Gallery_Match':
            return 'gallery'
        elif last == 'Probe':
            return 'probe'
        else:
            return 'distractor'

    def _make_dataset(self, mat_id_path):
        if mat_id_path is not None:
            # for gallery and probe
            annots = loadmat(mat_id_path)
            ids, _set = annots[self._dataset + '_ids'], annots[self._dataset + '_set']
            self._id_list = ids.reshape(-1)

            flatten_gallery_set = [_set[i, 0][0] for i in range(_set.shape[0])]
        else:
            # for distractor
            flatten_gallery_set = os.listdir(self._img_path)

        return flatten_gallery_set
    
   
    def __getitem__(self, index):
        img_name = self._samples[index]
        # img = PIL.Image.open(os.path.join(self._img_path, img_name[:-3]+'png'))
        img = PIL.Image.open(os.path.join(self._img_path, img_name[:-3]+'jpg'))
        return self.transform(img) , self.transform_flip(img)
     
    def __len__(self):
        return len(self._samples)
    
    
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


def calculate_acc(gallery_match_img_ID_pairs_path, probe_img_ID_pairs_path, gallery_feature_map, probe_feature_map, distractor_feature_map):
    
    # read in the ids
    gallery_ids = loadmat(gallery_match_img_ID_pairs_path)\
                            ['gallery_ids'] # (4443,1)
    probe_ids = loadmat(probe_img_ID_pairs_path)\
                            ['probe_ids'] # (3728,1)
    
    assert gallery_feature_map.shape[0] == 4443
    assert probe_feature_map.shape[0] == 3728
    # assert distractor_feature_map.shape[0] == 153428
                 
    assert gallery_ids.shape[0] == 4443
    assert probe_ids.shape[0] == 3728      


    gallery_feature_map = np.concatenate((gallery_feature_map, distractor_feature_map), axis=0)

    # concat the gallery id with the distractor id
    distractor_ids = -100 * np.ones((distractor_feature_map.shape[0], 1))
    gallery_ids = np.concatenate((gallery_ids, distractor_ids), axis=0) # (153428+4443, 1)
    
    # L2 distance
    dist = torch.cdist(torch.from_numpy(gallery_feature_map), \
                        torch.from_numpy(probe_feature_map)).numpy() # (157871,3728) = (#gallery, #probe)
    assert dist.shape == (157871, 3728)
    # cosine distance
    # dist = torch.matmul(torch.from_numpy(gallery_feature_map), torch.from_numpy(probe_feature_map).T).numpy()
        
    _CMC = np.zeros((probe_feature_map.shape[0], gallery_feature_map.shape[0])) # (3728,157871)
    ap = np.zeros((probe_feature_map.shape[0]))
    
    num_cores = multiprocessing.cpu_count()
        
    # x = Parallel(n_jobs=num_cores)(delayed(compute_AP)(np.where(gallery_ids == probe_ids[p,0])[0], np.argsort(dist[:, p])[::-1]) for p in range(probe_feature_map.shape[0]))
    x = Parallel(n_jobs=num_cores)(delayed(compute_AP)(np.where(gallery_ids == probe_ids[p,0])[0], np.argsort(dist[:, p])) for p in range(probe_feature_map.shape[0]))
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

def resnet_forward(ckp_path, loader_list, args):
    
    """ models """
    Resnet = get_model(args.network, dropout=0, fp16=False).cuda()
    #Resnet = iresnet34().cuda()
    #Resnet = IR_101((112, 112)).cuda()
    # Resnet = backbone_adaface.net.build_model('ir_101').cuda()
    # statedict = torch.load(ckp_path)['state_dict']
    # model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    # Resnet.load_state_dict(model_statedict)
    Resnet.load_state_dict(torch.load(ckp_path))
    #Resnet.load_state_dict(filter_module(torch.load(ckp_path)['Resnet']))
    # Resnet = backbone_magface.iresnet.iresnet100(
    #         pretrained=False,
    #         num_classes=512,
    #     ).cuda()
    # from collections import OrderedDict
    # def clean_dict_inf(model, state_dict):
    #     _state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         # # assert k[0:1] == 'features.module.'
    #         new_k = 'features.'+'.'.join(k.split('.')[2:])
    #         #print(k, new_k)
    #         if new_k in model.state_dict().keys() and \
    #         v.size() == model.state_dict()[new_k].size():
    #             _state_dict[new_k] = v
    #         # assert k[0:1] == 'module.features.'
    #         new_kk = '.'.join(k.split('.')[2:])
    #         #print(new_kk)
    #         if new_kk in model.state_dict().keys() and \
    #         v.size() == model.state_dict()[new_kk].size():
    #             _state_dict[new_kk] = v
    #     num_model = len(model.state_dict().keys())
    #     num_ckpt = len(_state_dict.keys())
    #     if num_model != num_ckpt:
    #         sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
    #             num_model, num_ckpt))
    #     return _state_dict
    # checkpoint = torch.load(ckp_path)
    # _state_dict = clean_dict_inf(Resnet, checkpoint['state_dict'])
    # model_dict = Resnet.state_dict()
    # model_dict.update(_state_dict)
    # Resnet.load_state_dict(model_dict)
    if (torch.cuda.device_count() > 1):
        Resnet = nn.DataParallel(Resnet)
    
        
    Resnet.eval()

    feature_list = []
    for loader in loader_list:
        with torch.no_grad():
            for i, (img, img_f) in enumerate(loader):
                # print(f'{i} / {len(loader)}')
                img = img.cuda()
                feat, norm, _ = Resnet(img)
                feat = feat * norm
                feat = feat.cpu().numpy()
                
                # including flip
                img_f = img_f.cuda()
                feat_f, norm_f, _ = Resnet(img_f)
                feat_f = feat_f * norm_f
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
                    
        feature_list.append(features)
        del features
    return feature_list


if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='r50')
    args = parser.parse_args()

    # dataset_root = "/mnt/HDD1/yuwei/dataset/Tinyface_we_aligned"
    dataset_root = "/mnt/HDD1/phudh/ICIP/insightface_lipface/data_test/aligned_pad_0.1_pad_high"
    gallery_match_img_ID_pairs_path = os.path.join(dataset_root, 'gallery_match_img_ID_pairs.mat')
    probe_img_ID_pairs_path = os.path.join(dataset_root, 'probe_img_ID_pairs.mat')

    gallery = Tinyface(img_path=os.path.join(dataset_root, 'Gallery_Match'),
                       mat_id_path=os.path.join(dataset_root, 'gallery_match_img_ID_pairs.mat'))

    gallery_db = DataLoader(dataset=gallery,
                            batch_size=512,
                            num_workers=12,
                            pin_memory=True,
                            shuffle=False)

    probe = Tinyface(img_path=os.path.join(dataset_root, 'Probe'),
                     mat_id_path=os.path.join(dataset_root, 'probe_img_ID_pairs.mat'))

    probe_db = DataLoader(dataset=probe,
                            batch_size=512,
                            num_workers=12,
                            pin_memory=True,
                            shuffle=False)

    gallery_distractor = Tinyface(img_path=os.path.join(dataset_root, 'Gallery_Distractor'))

    distractor_db = DataLoader(dataset=gallery_distractor,
                                batch_size=512,
                                num_workers=12,
                                pin_memory=True,
                                shuffle=False)
    
    loader_list = [gallery_db, probe_db, distractor_db]
    """ 
    ######################################
    ###  Resnet forward  #################
    ######################################
    
    """
    # print("#"*20 + " forwarding thru Resnet!  " + "#"*20)
    # root = "/mnt/data/chiawei/ResolutionInvariant_v2/ckps/L_arc_L_lip_L_lin/Resnet34_batchsz64_LR1e-05_m0.3_SGD_lamb_lip1_num_img_lip3_lip0.5_squaredTrue_lamb_lin80_shuffled_VGGface2_04_to_12_resize0.2/continue_09_to_11/"
    # ckp_path_list = os.listdir(root)
    # ckp_path_list = ["/mnt/data/chiawei/ResolutionInvariant_v3/ckps/L_arc_L_lin/Resnet34_batchsz64_LR1e-05_m0.3_SGD_lamb_lip0_num_img_lip3_lip0.5_squaredTrue_lamb_lin80_shuffled_VGGface2_04_to_11_resize0.2/e0_iter14706_loss_arcface2.12_loss_lips0.00_loss_lin0.06.pth",
    #                  "/mnt/data/chiawei/ResolutionInvariant_v3/ckps/L_arc_L_lin/Resnet34_batchsz64_LR1e-05_m0.3_SGD_lamb_lip0_num_img_lip3_lip0.5_squaredTrue_lamb_lin80_shuffled_VGGface2_04_to_11_resize0.2/e0_iter19608_loss_arcface0.70_loss_lips0.00_loss_lin0.05.pth",
    #                  "/mnt/data/chiawei/ResolutionInvariant_v3/ckps/L_arc_L_lin/Resnet34_batchsz64_LR1e-05_m0.3_SGD_lamb_lip0_num_img_lip3_lip0.5_squaredTrue_lamb_lin80_shuffled_VGGface2_04_to_11_resize0.2/e1_iter49020_loss_arcface1.93_loss_lips0.00_loss_lin0.04.pth",
    #                  "/mnt/data/chiawei/ResolutionInvariant_v3/ckps/L_arc_L_lin/Resnet34_batchsz64_LR1e-05_m0.3_SGD_lamb_lip0_num_img_lip3_lip0.5_squaredTrue_lamb_lin80_shuffled_VGGface2_04_to_11_resize0.2/e3_iter39216_loss_arcface1.19_loss_lips0.00_loss_lin0.05.pth"]
    # for i, ckp_path in enumerate(ckp_path_list):
    #     ckp_path_list[i] = os.path.join(root, ckp_path)
    # ckp_path_list = ["/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e0_iter35406_loss_arcface2.78_loss_lip0.000000.pth", # 68.59
                     # "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e1_iter35406_loss_arcface2.21_loss_lip0.000000.pth", # 67.84
                     # "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e2_iter35406_loss_arcface2.28_loss_lip0.000000.pth", # 67.89
                     # "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e3_iter35406_loss_arcface2.00_loss_lip0.000000.pth", # 66.22
                     # "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.1_squaredFalse_detach_HR_norm_resize0.2/e4_iter35406_loss_arcface1.71_loss_lip0.000000.pth"] # 66.01
    
    # ckp_path_list = [#"/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e0_iter35406_loss_arcface2.79_loss_lip0.001038.pth", # 68.21
                     #"/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e1_iter35406_loss_arcface2.26_loss_lip0.000892.pth", # 68.05
                     #"/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e2_iter35406_loss_arcface2.33_loss_lip0.001022.pth", # 68.18
                     # "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e3_iter35406_loss_arcface2.16_loss_lip0.000775.pth", # 67.94
                     # "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_detach_HR_norm_resize0.2/e4_iter35406_loss_arcface1.82_loss_lip0.001197.pth"] # 67.86
    # ckp_path_list = ["/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e0_iter35406_loss_arcface2.91_loss_lip0.000698.pth",
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e1_iter35406_loss_arcface2.42_loss_lip0.000384.pth",
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e2_iter35406_loss_arcface2.44_loss_lip0.000826.pth",
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e3_iter35406_loss_arcface2.37_loss_lip0.000619.pth",
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip1000_detach_HR_norm_resize0.2/e4_iter35406_loss_arcface1.98_loss_lip0.000468.pth"]
    
    # ckp_path_list = ["/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e0_iter35406_loss_arcface2.80_loss_lip0.001211.pth", # 68.16
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e1_iter35406_loss_arcface2.25_loss_lip0.000730.pth", # 68.13
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e2_iter35406_loss_arcface2.36_loss_lip0.001329.pth", # 68.43
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e3_iter35406_loss_arcface2.16_loss_lip0.000850.pth", # 68.16
    #                  "/mnt/data/chiawei/ResolutionInvariant_v4_lipschitz_cosine/ckps/L_arc_L_lip/ms1mv3_arcface_r34_batchsz128_m0.5_LR0.0001_num_img_lip1_lip0.01_squaredFalse_lamb_lip100_detach_HRFalse_resize0.2/e4_iter35406_loss_arcface1.84_loss_lip0.000794.pth"] # 67.95
    
    ckp_path_list = [
               "/mnt/HDD1/phudh/ICIP/insightface_lipface/work_dirs/ms1mv2_r50_lip_ngpu2_p05_lrate001_lr42_MixDegraded_attn_8epoch/model_e7.pt",
                    "/mnt/HDD1/phudh/ICIP/insightface_lipface/work_dirs/ms1mv2_r50_lip_ngpu2_p05_lrate001_lr42_originalDegraded_attn/model_e5.pt"

                     ]
    # ckp_path_list.sort(key=lambda x: os.path.getmtime(x))
    # ckp_path_list.sort()
    
    for ckp_path in ckp_path_list:
        # if ckp_path[:2] != 'e4':
        #     continue
        if (os.path.basename(ckp_path) == 'runs'):
            continue
        # ckp_path = os.path.join(root, ckp_path)
        print(ckp_path)
    
        gallery_feat, probe_feat, distractor_feat = resnet_forward(ckp_path, loader_list, args)
        
        gallery_feat -= np.mean(gallery_feat, axis=0)
        probe_feat -= np.mean(probe_feat, axis=0)
        distractor_feat -= np.mean(distractor_feat, axis=0)
        
        gallery_feat = normalize(gallery_feat)
        probe_feat = normalize(probe_feat)
        distractor_feat = normalize(distractor_feat)
        # print("#"*20 + " calculating acc!  " + "#"*20)
        
        mAP, CMC = calculate_acc(gallery_match_img_ID_pairs_path,
                                 probe_img_ID_pairs_path,
                                 gallery_feat,
                                 probe_feat,
                                 distractor_feat)
        # print(f'mAP = {mAP}')
        print(f'r1 precision = {CMC[0]}')
        # print(f'r5 precision = {CMC[4]}')
        # print(f'r10 precision = {CMC[9]}')
        # print(f'r20 precision = {CMC[19]}')