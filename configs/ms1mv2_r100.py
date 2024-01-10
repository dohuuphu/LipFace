from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

# 
config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r100"
config.resume = False
config.output = "/mnt/HDD1/yuwei/insightface_lipface/work_dirs/ms1mv3_r100_lip_finetune_no_adapt/"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/home/RI/dataset/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

# lipschitz
config.lamb_lip = 800
config.num_img_lip = 1
config.lip = 0.01
config.squared = False
config.detach_HR = False
config.p = 0.2
config.margin_lip = 0.0
config.tao = 2.0
config.alpha = 0.99