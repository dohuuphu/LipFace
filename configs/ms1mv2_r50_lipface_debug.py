from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = True
config.pretrained = "work_dirs/ms1mv3_r50_pre/backbone.pth"
config.output = "work_dirs/ms1mv2_r50_lip_ngpu2_p05_lrate001_lr42_debug/"
config.ckpt_dir = "work_dirs/ms1mv2_r50_arc_ngpu2/"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.01
config.verbose = 2000
config.dali = False
config.save_all_states = True

config.rec = "data_test/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 6
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

# lipschitz
config.lamb_lip = 800
config.num_img_lip = 1
config.lip = 0.01
config.squared = False
config.detach_HR = False
config.p = 0.5
config.lamb_lip_negative = 800
config.margin_lip = 0.0
config.tao = 2.0
config.alpha = 0.99