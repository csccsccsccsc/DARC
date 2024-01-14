import os
import importlib
from load_save_model import save_model
from train_wrap_nepoch import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

# from networks.networks_2d.deeplabv3_baseline import DeepLab as build_model_func
# from networks.networks_2d.deeplabv3_baseline import DeepLab as build_model_func
# 

# from networks.networks_2d.unet_gray_branch_resort import UNet as build_model_func
from networks.networks_2d.unet_pred_ratio_all_rescaling_gray_resort_final import UNet as build_model_func
from networks.norm_blocks.IN_MLP_final import IN_MLP_Rescaling4_detach_small as IN_MLP_Rescaling

# from networks.bns.dsbn import DomainSpecificBatchNorm

from dataset.pathology.pathology_rgb2gray_inputGT2 import getTestDataLoaders as dataloader
import random
import numpy as np
import tqdm
import skimage
import skimage.io
import skimage.morphology
import skimage.measure
import scipy
import scipy.ndimage
import cv2

import logging
import logging.config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(4)

from metrics_aji import get_fast_aji, get_fast_pq, get_dice_1, remap_label

import warnings
warnings.filterwarnings('ignore') 

#################################################
##### Visualization

import random
valid_colors = [\
                    [0,255,255], [0, 255, 127], [255, 255, 0], \
                    [255, 99, 71], [255, 0, 0], [238, 130, 238], \
                    [131, 111, 255], [0, 0, 255], [0, 191, 255], \
                    [0, 255, 0], [179, 238, 58], [255, 215, 0], \
                    [255, 130, 71], [255, 165, 0], [255, 0, 255], \
                    [155, 48, 255],\
               ]

def coloring_inst(label, color_list=valid_colors):
    ncolor = len(color_list)
    cset = np.unique(label[label>0])
    color_inst = np.zeros([3,]+list(label.shape), dtype=np.float32)
    cmap = []
    for ic in cset:
        iccolor = random.randint(0, ncolor-1)
        color_inst[0][label == ic] = valid_colors[iccolor][0]
        color_inst[1][label == ic] = valid_colors[iccolor][1]
        color_inst[2][label == ic] = valid_colors[iccolor][2]
        cmap.append(iccolor)
    color_inst = np.uint8(np.transpose(color_inst, [1,2,0]))
    return color_inst, cmap

def draw_contour(label, dilation_iter=1, erosion_iter=1, color=[224, 16, 16]):
    bnd = np.ones(list(label.shape) + [3, ], dtype=np.uint8) * 255
    cset = np.unique(label[label>0])
    for ic in cset:
        icmap = label==ic
        if dilation_iter >0:
            d_icmap = scipy.ndimage.morphology.binary_dilation(icmap, iterations=dilation_iter).astype(np.float32)
        else:
            d_icmap = icmap.astype(np.float32)
        if erosion_iter >0:
            e_icmap = scipy.ndimage.morphology.binary_erosion(icmap, iterations=erosion_iter).astype(np.float32)
        else:
            e_icmap = icmap.astype(np.float32)
        cur_bnd = d_icmap - e_icmap
        bnd[cur_bnd>0, 0] = color[0]
        bnd[cur_bnd>0, 1] = color[1]
        bnd[cur_bnd>0, 2] = color[2] 
    return bnd


#################################################
shared_log_config = {
    "version": 1,
    "formatters": {
        "default": {
            'format':'%(asctime)s %(filename)s %(lineno)s %(levelname)s %(message)s',
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "file":{
            "class": "logging.FileHandler",
            "level":20,
            "filename": None,
            "formatter": "default",
        }
    },
    "loggers": {
        "console_logger": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "file_logger":{
            "handlers": ["file"],
            "level": "INFO",
            "propagate": False,
        }
    },
    "disable_existing_loggers": True,
}
#################################################
#
def post_processing_segbnd(seg, bnd, prob_thres=0.5, dilation=2):
    instance = (seg-bnd)>=prob_thres
    lbl_center = scipy.ndimage.label(instance)[0]
    cset = np.unique(lbl_center[lbl_center>0])
    lbl = np.zeros(lbl_center.shape, dtype=np.int16)
    for ic in cset:
        icmap = lbl_center==ic
        if dilation >0:
            icmap = scipy.ndimage.morphology.binary_dilation(icmap, iterations=dilation)
        icmap[seg<prob_thres] = False
        lbl[icmap] = ic
    return lbl

def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def kfold_list(n, K, kfold_split_seed=123):
    rnd_state = np.random.RandomState(kfold_split_seed)
    dataset_rnd_sort = np.array(list(range(n)), dtype=np.int16)
    rnd_state.shuffle(dataset_rnd_sort)
    idxlist = list(range(0, n, 1))
    val_idx = [list(range(i*(n//K), (i+1)*(n//K))) for i in range(0, K, 1)]
    train_idx = [list(range(0, i*int(n//K), 1))+list(range((i+1)*(n//K), n, 1)) for i in range(0, K, 1)]
    if val_idx[-1][-1] < n-1:
        val_idx[-1] = val_idx[-1]+list(range(val_idx[-1][-1]+1, n, 1))
        train_idx[-1] = list(range(0, (K-1)*(n//K)))
    val_from_list = []
    train_from_list = []
    for ival, itrain in zip(val_idx, train_idx):
        val_from_list.append(dataset_rnd_sort[ival])
        train_from_list.append(dataset_rnd_sort[itrain])
    return train_from_list, val_from_list

def filter_img(s):
    if s[-4:] in ['.jpg', '.png', '.bmp', '.tif']:
        return True
    else:
        return False

def get_largest_fillhole(binary):
    label_image = skimage.measure.label(binary)
    regions = skimage.measure.regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


print('-----'*30)

print('Working dir', os.getcwd())

IN_CHANNELS = 3
LOAD_CHECKPOINT = False
LOG_DIRECTORY = None
nd_features = 64
seeds = [12345, ]


Data_Root_PATH = ''
# DATASET_NAME = ['CoNSeP', 'cpm17', 'MoNuSeg', 'pannuke', 'deepliif', 'bc_deepliif']
# All_Train_Split_IMG = ['/data2/cong/dataset/CoNSeP/Train/Images', '/data2/cong/dataset/cpm17/train/Images', '/data/cong/datasets/kumar/monuseg/MoNuSegTrainData/images', '/data2/cong/dataset/pannuke/reorganized_dataset/fold_1/images', '/data2/cong/dataset/deepliif/deepliif/training/IHC', '/data2/cong/dataset/deepliif/bc_deepliif/training/IHC']
# All_Train_Split_MSK = ['/data2/cong/dataset/CoNSeP/Train/Labels', '/data2/cong/dataset/cpm17/train/Labels', '/data/cong/datasets/kumar/monuseg/MoNuSegTrainData/masks', '/data2/cong/dataset/pannuke/reorganized_dataset/fold_1/inst_maps', '/data2/cong/dataset/deepliif/deepliif/training/Seg', '/data2/cong/dataset/deepliif/bc_deepliif/training/Seg']
# All_Test_Split_IMG = ['/data2/cong/dataset/CoNSeP/Test/Images', '/data2/cong/dataset/cpm17/test/Images', '/data/cong/datasets/kumar/monuseg/test/images', '/data2/cong/dataset/pannuke/reorganized_dataset/fold_3/images', '/data2/cong/dataset/deepliif/deepliif/validation/IHC', '/data2/cong/dataset/deepliif/bc_deepliif/validation/IHC']
# All_Test_Split_MSK = ['/data2/cong/dataset/CoNSeP/Test/Labels', '/data2/cong/dataset/cpm17/test/Labels', '/data/cong/datasets/kumar/monuseg/test/masks', '/data2/cong/dataset/pannuke/reorganized_dataset/fold_3/inst_maps', '/data2/cong/dataset/deepliif/deepliif/validation/Seg', '/data2/cong/dataset/deepliif/bc_deepliif/validation/Seg']

DATASET_NAME = ['CoNSeP', 'cpm17', 'deepliif', 'bc_deepliif']
All_Train_Split_IMG = ['path_to_datasets/CoNSeP/Train/Images', 'path_to_datasets/cpm17/train/Images', 'path_to_datasets/deepliif/deepliif/training/IHC', 'path_to_datasets/deepliif/bc_deepliif/training/IHC', ]
All_Train_Split_MSK = ['path_to_datasets/CoNSeP/Train/Labels', 'path_to_datasets/cpm17/train/Labels', 'path_to_datasets/deepliif/deepliif/training/Seg', 'path_to_datasets/deepliif/bc_deepliif/training/Seg', ]
All_Test_Split_IMG = ['path_to_datasets/CoNSeP/Test/Images', 'path_to_datasets/cpm17/test/Images', 'path_to_datasets/deepliif/deepliif/validation/IHC', 'path_to_datasets/deepliif/bc_deepliif/validation/IHC', ]
All_Test_Split_MSK = ['path_to_datasets/CoNSeP/Test/Labels', 'path_to_datasets/cpm17/test/Labels', 'path_to_datasets/deepliif/deepliif/validation/Seg', 'path_to_datasets/deepliif/bc_deepliif/validation/Seg', ]


subdir = ''
basic_dataset = 'pathology'
ori_train_mode = 'Basic' + '_' + 'rnd_pad' + '' + ''
valid_domain = [0, 1, 2, 3]
norm_name_list = ['DARC', ] 
norm_module_list = [IN_MLP_Rescaling, ]

for train_mode in [ori_train_mode+'_', ]:

    load_ema = False
    visualize = False
    cal_type = ['aji', 'pq', 'dice']
    assert(len(cal_type)>0)
    for _e in cal_type:
        assert(_e in ['aji', 'pq', 'dice'])

    backbone_type = 'mobilenet'
    out_stride = 16
    resz = None
    batch_size = 1


    save_root = os.path.join('./predictions')
    os.makedirs(save_root, exist_ok=True)

    patch_size = 224
    valid_size = patch_size//2

    patch_valid_s = (patch_size-valid_size) // 2
    patch_valid_e = patch_valid_s + valid_size

    res_suffix = '_patch_pred' + '' + ''
    if not (len(cal_type) == 1 and cal_type[0] == 'aji'):
        res_suffix += '_' + str(cal_type)


    for norm_name, norm_module in zip(norm_name_list, norm_module_list):

        model_name_suffix = 'baseline' + '_' + norm_name + ''

        for irnd, seed in enumerate(seeds):

            for isplit, train_dataset_name in enumerate(DATASET_NAME):

                if not isplit in valid_domain:
                    continue
                
                dataset = basic_dataset + '_' + train_dataset_name

                def make_dsbn(nc):
                    return DomainSpecificBatchNorm(num_features=nc, num_classes=len(cur_train_split))

                print(train_dataset_name)

                random.seed(seed)
                np.random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                torch.cuda.manual_seed_all(seed)
                torch.manual_seed(seed)

                print('-----'*10)
                model_name = 'ResUnet_' + model_name_suffix + '_'+ backbone_type + '_rnd'+str(irnd)+'_split'+str(isplit)

                os.makedirs('./res', exist_ok=True)

                log_file_path = './res/' + '_'.join([model_name, train_mode, dataset])+res_suffix + '.log'
                cur_log_config = shared_log_config
                cur_log_config["handlers"]["file"]["filename"] = log_file_path
                logging.config.dictConfig(cur_log_config)
                logger_console = logging.getLogger("console_logger")
                logger_file = logging.getLogger("file_logger")

                for cur_domain_idx, (image_root, label_root) in enumerate(zip(All_Test_Split_IMG, All_Test_Split_MSK)):
                    if not cur_domain_idx in valid_domain:
                        continue
                    # if train_dataset_name != DATASET_NAME[cur_domain_idx]:
                    #     continue
                    
                    # if cur_domain_idx not in [4,5]:
                    #     continue

                    if visualize:
                        os.makedirs(os.path.join(save_root, '_'.join([model_name, train_mode, dataset])+res_suffix), exist_ok=True)

                    name_list = list(filter(filter_img, os.listdir(image_root)))
                    name_list.sort()

                    Testloader = dataloader(image_root, label_root, batch_size=batch_size, name_list=name_list, train_crop=None, test_crop=None, resz=resz, getbnd_preload=True)

                    with torch.no_grad():
                        
                        # model = build_model_func(num_classes=2, backbone=backbone_type, output_stride=out_stride, Norm=norm_module)
                        if backbone_type == 'ResUNet':
                            model = build_model_func(n_channels=3, n_features=32, n_out=2, norm=norm_module)
                        else:
                            # model = build_model_func(num_classes=2, backbone=backbone_type, output_stride=out_stride, Norm=norm_module, )
                            model = build_model_func(n_channels=3, n_features=32, n_out=2, norm=norm_module)
                        for m in model.modules():
                            if isinstance(m, DomainSpecificBatchNorm):
                                m.avg_all = True
                        model = model.cuda()

                        for ickp in [100, ]:
                            print(model_name+'_'+train_mode+'_'+dataset, ickp)

                            if load_ema:
                                weights = torch.load(os.path.join('./CHECKPOINT', 'checkpoint_'+model_name+'_'+train_mode+'_'+dataset, 'CHECKPOINT_epoch_'+str(ickp)+'.t7'))
                                weights_to_load = dict()
                                for k, v in weights.items():
                                    if k.find('s_model') >= 0:
                                        k2 = k.replace('s_model.', '')
                                        weights_to_load.update({k2:v})
                                model.load_state_dict(weights_to_load)
                            else:
                                load_info = model.load_state_dict(\
                                    torch.load(os.path.join('./CHECKPOINT', 'checkpoint_'+model_name+'_'+train_mode+'_'+dataset, 'CHECKPOINT_epoch_'+str(ickp)+'.t7')),\
                                    strict = True
                                )
                                print(load_info)
                            model.eval()

                            if hasattr(model, 'amp_norm') and model.amp_norm is not None:
                                model.amp_norm.fix_amp = True
                                print('Fix AMP')

                            aji_list = []
                            pq_list = []
                            sq_list = []
                            dq_list = []
                            dice_list = []

                            qbar = tqdm.tqdm(enumerate(Testloader))
                            for i, batch in qbar:

                                image = batch[0]
                                padding_sz = batch[2]
                                target_seg = batch[3]
                                ori_image = batch[4][0]

                                # if cur_test_split[0] == 'Domain4/ALL':
                                #     image = image[:, :, 144:(144+512), 144:(144+512)]
                                #     target_seg = target_seg[:, :, 144:(144+512), 144:(144+512)]
                                image_input = torch.FloatTensor(image)
                                b,c,h,w = image_input.shape

                                h2, w2 = int(valid_size*np.ceil(h/valid_size)), int(valid_size*np.ceil(w/valid_size))
                                h3 = h2+(patch_size-valid_size)
                                w3 = w2+(patch_size-valid_size)
                                ph = h3-h
                                pw = w3-w
                                tph = ph//2
                                bph = ph-tph
                                lpw = pw//2
                                rpw = pw-pw//2
                                image_input = F.pad(image_input, (lpw, rpw, tph, bph), mode='reflect')
                                image_patches = image_input.unfold(2, patch_size, valid_size).unfold(3, patch_size, valid_size)
                                image_patches_shape = image_patches.shape
                                image_patches = image_patches.permute(0, 2, 3, 1, 4, 5).contiguous().flatten(start_dim=0, end_dim=2)
                                n_patches = image_patches_shape[0] * image_patches_shape[2] * image_patches_shape[3]
                                predictions = []
                                subbatch_size = min(16, n_patches)
                                for sidx_subbatch in range(0, n_patches, subbatch_size):
                                    eidx_subbatch = min(sidx_subbatch+subbatch_size, n_patches)
                                    subbatch = image_patches[sidx_subbatch:eidx_subbatch].cuda()
                                    subbatch_size_real = subbatch.shape[0]
                                    subbatch_predictions = model(subbatch)[0]
                                    subbatch_predictions_pred = subbatch_predictions.data.cpu()[0:subbatch_size_real, :, patch_valid_s:patch_valid_e, patch_valid_s:patch_valid_e]
                                    predictions.append( subbatch_predictions_pred )
                                predictions = torch.cat(predictions, dim=0).view(image_patches_shape[0], image_patches_shape[2], image_patches_shape[3], 2, valid_size, valid_size)
                                predictions = torch.permute(predictions, (0,3,1,4,2,5)).contiguous().view(b,2,valid_size*image_patches_shape[2],valid_size*image_patches_shape[3])
                                predictions = predictions[:, :, (tph-(patch_size-valid_size)//2):(tph-(patch_size-valid_size)//2)+h, (lpw-(patch_size-valid_size)//2):(lpw-(patch_size-valid_size)//2)+w]

                                for i_inbatch in range(b):

                                    cur_name = name_list[i*batch_size+i_inbatch]
                                    cur_name = cur_name[0:-4]

                                    i_target_seg = target_seg[i_inbatch]
                                    h0, w0 = i_target_seg.shape
                                    i_pred_seg = np.transpose(predictions[i_inbatch, :, 0:int(h-padding_sz[i_inbatch][0]), 0:int(w-padding_sz[i_inbatch][1])], [1,2,0])
                                    if resz is not None:
                                        i_pred_seg = cv2.resize(i_pred_seg, (w0, h0))

                                    i_pred_inst = post_processing_segbnd(i_pred_seg[:, :, 0], i_pred_seg[:, :, 1])

                                    try:
                                        i_target_seg = remap_label(i_target_seg)
                                        i_pred_inst = remap_label(i_pred_inst)
                                    except:
                                        pass

                                    if 'aji' in cal_type:
                                        try:
                                            aji = get_fast_aji(i_target_seg, i_pred_inst)
                                        except:
                                            if (i_target_seg>0).any() or (i_pred_inst>0).any():
                                                aji = 0.0
                                            else:
                                                aji = 1.0
                                    else:
                                        aji = -1
                                    aji_list.append(aji)

                                    if 'pq' in cal_type:
                                        try:
                                            dq, sq, pq = get_fast_pq(i_target_seg, i_pred_inst)[0]
                                        except:
                                            if (i_target_seg>0).any() or (i_pred_inst>0).any():
                                                dq, sq, pq = 0., 0., 0.
                                            else:
                                                dq, sq, pq = 1., 1., 1.
                                    else:
                                        dq, sq, pq = -1, -1, -1
                                    pq_list.append(pq)
                                    dq_list.append(dq)
                                    sq_list.append(sq)

                                    if 'dice' in cal_type:
                                        dice = get_dice_1(i_target_seg, i_pred_inst)
                                    else:
                                        dice = -1
                                    dice_list.append(dice)


                                    qbar.set_description('{:d} : {:.5f}'.format(i, aji, ))

                                    if visualize:
                                        ori_image = ori_image[:,:,:3]
                                        image2show_t = [ori_image, coloring_inst(i_target_seg)[0], ]
                                        image2show_t.append(cv2.cvtColor(np.uint8((i_target_seg>0)*255), cv2.COLOR_GRAY2RGB))
                                        t_bnd = draw_contour(i_target_seg)
                                        image2show_t.append(t_bnd)
                                        image2show_t.append(t_bnd)
                                        # for tmp in image2show_t:
                                        #     print(tmp.shape)
                                        image2show_t = np.concatenate(image2show_t, axis=1)

                                        image2show_b = [ori_image, coloring_inst(i_pred_inst)[0], ]
                                        image2show_b.append(cv2.cvtColor(np.uint8(i_pred_seg[:, :, 0]*255), cv2.COLOR_GRAY2RGB))
                                        image2show_b.append(cv2.cvtColor(np.uint8(i_pred_seg[:, :, 1]*255), cv2.COLOR_GRAY2RGB))
                                        image2show_b.append(draw_contour(i_pred_inst))
                                        image2show_b = np.concatenate(image2show_b, axis=1)

                                        image2show = np.uint8(np.concatenate((image2show_t, image2show_b), axis=0))

                                        skimage.io.imsave(os.path.join(save_root, '_'.join([model_name, train_mode, dataset])+res_suffix, 'tr_'+train_dataset_name+'_te_'+DATASET_NAME[cur_domain_idx]+'_'+cur_name+'_{:.5f}'.format(aji, )+'.png'), \
                                            image2show, check_contrast=False)

                            # cur_info = 'Train Domain: (' + train_dataset_name + ') '+ 'Test Domain: (' + DATASET_NAME[cur_domain_idx] + ') '+ image_root + '; Epoch: '+ str(ickp) + '; AJI: {:.5f}'.format(np.mean(aji_list))
                            # logger_console.info(cur_info)
                            # logger_file.info(cur_info)

                            if 'aji' in cal_type:
                                cur_info = 'Train Domain: (' + train_dataset_name + ') '+ 'Test Domain: (' + DATASET_NAME[cur_domain_idx] + ') '+ image_root + '; Epoch: '+ str(ickp) + \
                                            '; AJI: {:.5f}'.format(np.mean(aji_list))
                                logger_console.info(cur_info)
                                logger_file.info(cur_info)

                            if 'pq' in cal_type:
                                cur_info = 'Train Domain: (' + train_dataset_name + ') '+ 'Test Domain: (' + DATASET_NAME[cur_domain_idx] + ') '+ image_root + '; Epoch: '+ str(ickp) + \
                                            '; PQ: {:.5f}, SQ: {:.5f}, DQ: {:.5f}'.format(np.mean(pq_list), np.mean(sq_list), np.mean(dq_list))
                                logger_console.info(cur_info)
                                logger_file.info(cur_info)

                            if 'dice' in cal_type:
                                cur_info = 'Train Domain: (' + train_dataset_name + ') '+ 'Test Domain: (' + DATASET_NAME[cur_domain_idx] + ') '+ image_root + '; Epoch: '+ str(ickp) + \
                                            '; DICE: {:.5f}'.format(np.mean(dice_list))
                                logger_console.info(cur_info)
                                logger_file.info(cur_info)

