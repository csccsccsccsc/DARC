import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR

from networks.networks_2d.unet_pred_ratio_all_rescaling_gray_resort_final import UNet as build_model_func


# from networks.networks_2d.deeplabv3_baseline import DeepLab as build_model_func

# from networks.networks_2d.unet import make_gn

from loss_wrap import SegLossDSCRatio as lossfunc
from load_save_model import save_model
from train_wrap_nepoch import Trainer


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.set_num_threads(8)

def filter_img(s):
    if s[-4:] in ['.jpg', '.png', '.bmp', '.tif']:
        return True
    else:
        return False

print('-----'*30)

print('Working dir', os.getcwd())

LOAD_CHECKPOINT = False
LOG_DIRECTORY = None
seeds = [12345, 23451, 34512, 45123, 51234]

Data_Root_PATH = ''

DATASET_NAME = ['CoNSeP', 'cpm17', 'deepliif', 'bc_deepliif']
All_Train_Split_IMG = ['path_to_datasets/CoNSeP/Train/Images', 'path_to_datasets/cpm17/train/Images', 'path_to_datasets/deepliif/deepliif/training/IHC', 'path_to_datasets/deepliif/bc_deepliif/training/IHC', ]
All_Train_Split_MSK = ['path_to_datasets/CoNSeP/Train/Labels', 'path_to_datasets/cpm17/train/Labels', 'path_to_datasets/deepliif/deepliif/training/Seg', 'path_to_datasets/deepliif/bc_deepliif/training/Seg', ]
All_Test_Split_IMG = ['path_to_datasets/CoNSeP/Test/Images', 'path_to_datasets/cpm17/test/Images', 'path_to_datasets/deepliif/deepliif/validation/IHC', 'path_to_datasets/deepliif/bc_deepliif/validation/IHC', ]
All_Test_Split_MSK = ['path_to_datasets/CoNSeP/Test/Labels', 'path_to_datasets/cpm17/test/Labels', 'path_to_datasets/deepliif/deepliif/validation/Seg', 'path_to_datasets/deepliif/bc_deepliif/validation/Seg', ]

subdir = ''
basic_dataset = 'pathology'
basic_train_mode = 'Basic' + '_' + 'rnd_pad' + ''

batch_size = 4
n_iter = 1000
max_epoch = 100
save_every = max_epoch//10
step_size = max_epoch//10
milestones = np.arange(step_size, max_epoch, step_size)
init_lr = 1e-3
end_lr = 1e-5
gamma = np.power(end_lr / init_lr, 1.0/len(milestones))
print(step_size, milestones, gamma)
out_stride = 16

remove_checkpoints = False

os.makedirs('./logs', exist_ok=True)

norm_name = 'DARC'


model_name_suffix = 'baseline' + '_' + norm_name + ''

for irnd, seed in enumerate(seeds):

    for isplit, (cur_dataset_name, cur_train_split_img, cur_train_split_msk,) in enumerate(zip(DATASET_NAME, All_Train_Split_IMG, All_Train_Split_MSK,)):

        for mode in ['', ]:

            if len(mode) == 0:
                from dataset.pathology.pathology_rgb2gray_inputGT import getTrainDataLoaders as dataloader
            elif mode == 'pred_finetune':
                from dataset.pathology.pathology_rgb2gray import getTrainDataLoaders as dataloader
                
            train_mode = basic_train_mode + '_' + mode
        
            random.seed(seed)
            np.random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            print('-----'*10)

            backbone_type = 'mobilenet'
            model_name = 'ResUnet_' + model_name_suffix + '_'+ backbone_type + '_rnd'+str(irnd)+'_split'+str(isplit)
            dataset = basic_dataset + '_' + cur_dataset_name

            image_root = [cur_train_split_img, ]
            label_root = [cur_train_split_msk, ]

            # No validation subset
            train_name_list = []
            val_name_list = []
            for i_root in range(len(image_root)):
                valid_names = list(filter(filter_img, os.listdir(image_root[i_root])))
                train_name_list.append(valid_names)
                val_name_list.append(valid_names)
                print(image_root[i_root], len(valid_names))
            if os.path.exists(os.path.join('./CHECKPOINT', 'checkpoint_'+model_name+'_'+train_mode+'_'+dataset, 'CHECKPOINT_epoch_'+str(max_epoch)+'.t7')):
                print('SKIP: ' + os.path.join('./CHECKPOINT', 'checkpoint_'+model_name+'_'+train_mode+'_'+dataset, 'CHECKPOINT_epoch_'+str(max_epoch)+'.t7'))
                continue

            Trainloader = dataloader(image_root, label_root, batch_size=batch_size, name_list=[train_name_list, val_name_list], \
                                    train_crop=224, test_crop=None, resz=None, aug=True, rnd_pad=False, \
                                    domain_balanced=True, num_samples=n_iter*batch_size, \
                                    freq_aug='none', other_domain_root=None, )

            print('Train Baches: ', len(Trainloader))

            print('model=' + model_name)
            print('dataset=' + dataset)

            kwargs={}
            additional_notes= ''
            kwargs['additional_notes'] = additional_notes
            SAVE_PATH = os.getcwd()+'/'+dataset+'/'+train_mode+'_'+model_name+'/'
            kwargs['save_path'] = SAVE_PATH
            RESULTS_DIRECTORY = os.getcwd()+'/'+dataset+'/'+train_mode+'_'+model_name+'/plots/'

            model = build_model_func(n_channels=3, n_features=32, n_out=2)
            model = model.cuda()

            loss = lossfunc()

            LOG_DIRECTORY = './logs/' + '_'.join([model_name, dataset, train_mode])+'.txt'

            trainer = Trainer(loss, None, LOG_DIRECTORY, valid_in_train=False, save_every=save_every)

            if LOAD_CHECKPOINT:
                print('Not Implement')

            optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.99))
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

            if not isinstance(scheduler, ReduceLROnPlateau):
                assert(not remove_checkpoints) # When using other schedulers, should keep the checkpoints

            print ('Starting Training')
            trainloss_to_file,testloss_to_file,trainMetric_to_file,testMetric_to_file,Parameters= trainer.Train(model,optimizer,
                                                                                                Trainloader, None, epochs=max_epoch,Train_mode=train_mode,
                                                                                                Model_name=model_name,
                                                                                                Dataset=dataset, scheduler=scheduler)
