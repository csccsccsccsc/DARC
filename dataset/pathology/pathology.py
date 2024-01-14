import os
import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler, WeightedRandomSampler
from prefetch_generator import BackgroundGenerator

import skimage
from skimage import io
from skimage.transform import resize
from skimage.morphology import remove_small_objects
from skimage import exposure
from skimage.color import rgb2hed, hed2rgb

import scipy.io as scio
import numpy as np
import random
import scipy
import scipy.io
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion, generate_binary_structure
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import stats
import cv2
import tqdm
from stardist import star_dist

from dataset.pathology.normalizeStaining import normalizeStaining, estimateHE



def GetBoundary(mask, width=5, class_axis=0, merge='sum', nclass=2, dwidth=-1, ewidth=-1,):
    if len(mask.shape) == 3:
        masks = np.split(mask, nclass, axis=class_axis)
    else:
        masks = [mask, ]
    if merge == 'sum':
        bnd = np.zeros(masks[0].shape, dtype=np.float32)
    else:
        bnd = []
    if dwidth < 0 or dwidth is None:
        dwidth = width
    if ewidth < 0 or ewidth is None:
        ewidth = width
    for ic_mask in masks:
        ic_mask = ic_mask >= 0.5
        if dwidth > 0:
            d_ic_mask = ndimage.binary_dilation(ic_mask, iterations=dwidth)
        if ewidth > 0:
            e_ic_mask = ndimage.binary_erosion(ic_mask, iterations=ewidth)
        if merge == 'sum':
            bnd += np.float32(np.logical_xor(d_ic_mask, e_ic_mask))
        else:
            bnd.append(np.float32(np.logical_xor(d_ic_mask, e_ic_mask)))
    if merge == 'sum':
        bnd = np.float32(bnd > 0)
    else:
        bnd = np.stack(bnd, axis=class_axis)
    return bnd



def filter_img(s):
    if s[-4:] in ['.jpg', '.png', '.bmp', '.tif']:
        return True
    else:
        return False

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def amp_swap( source_amp, target_amp, L=0.1 , ratio=0):

    # source: H, W, _
    # target: H, W, _
    
    source_amp = np.fft.fftshift( source_amp, axes=(0, 1) )
    target_amp = np.fft.fftshift( target_amp, axes=(0, 1) )

    h, w = source_amp.shape[0:2]
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    source_amp[h1:h2, w1:w2, :] = source_amp[h1:h2, w1:w2, :] * ratio + target_amp[h1:h2, w1:w2, :] * (1- ratio)
    source_amp = np.fft.ifftshift( source_amp, axes=(0, 1) )

    return source_amp

def freq_swap_aug(source_img, target_img, L=0.1 , ratio=0):
    ori_shape = source_img.shape
    h, w = ori_shape[0:2]
    ht, wt = target_img.shape[0:2]
    if ht!=h or wt!=w:
        target_img = cv2.resize(target_img, (w, h))

    if len(source_img.shape) == 2:
        source_img = np.expand_dims(source_img, axis=2)
    if len(target_img.shape) == 2:
        target_img = np.expand_dims(target_img, axis=2)


    source_fft = np.fft.fft2(source_img, axes=(0, 1))
    source_amp, source_ang = np.abs(source_fft), np.angle(source_fft)

    target_fft = np.fft.fft2(target_img, axes=(0, 1))
    target_amp = np.abs(target_fft)

    source_amp_aug = amp_swap( source_amp, target_amp, L=L, ratio=ratio)

    source_fft_aug = source_amp_aug * np.exp( 1j * source_ang )

    source_img_aug = np.fft.ifft2( source_fft_aug, axes=(0, 1) )
    source_img_aug = np.real(source_img_aug).reshape(ori_shape)


    return source_img_aug

# ImageNet
norm_mean=np.array([[[0.485, 0.456, 0.406]]])
norm_std=np.array([[[0.229, 0.224, 0.225]]])

HERef = np.array([[0.5626, 0.2159],
                  [0.7201, 0.8012],
                  [0.4062, 0.5581]])
maxCRef = np.array([1.9705, 1.0308])


class PathologyDataset(Dataset):
    def __init__(self, image_root, label_root, name_list=None, if_training=False, crop=None, resz=None, stain_norm=False, division=32, \
                domain_label=0, domain_confidence=None, aug=False, \
                return_domain_lbl=False, return_domain_confidence=False, return_tgt_image=False, \
                gamma_aug=True, with_canny=False, rnd_pad=False, \
                freq_aug=None, other_domain_root=None, p_freq_aug=0.5, bnd_width_freq_aug=0.1, max_rnd_resz=1.0, \
                stain_aug = False, stain_mix=False, return_he=False, size_aug=False, \
                return_contour=False, getbnd_online=False, return_dist=False, n_dist=8, \
                hist_aug=False, hist_aug_extra_path=None, \
        ):

        self.image_root = image_root
        self.label_root = label_root
        if name_list is not None:
            self.name_list = name_list
        else:
            self.name_list = os.listdir(self.image_root)
            self.name_list.sort()

        self.if_training = if_training
        self.resz = resz
        self.crop = crop
        self.division = division
        if if_training:
            self.mode = 'train'
        else:
            self.mode = 'valid'

        self.return_contour = return_contour
        self.return_domain_lbl = return_domain_lbl
        self.return_domain_confidence = return_domain_confidence
        self.return_tgt_image = return_tgt_image
        self.return_he = return_he

        self.aug = aug
        self.gamma_aug = gamma_aug
        self.freq_aug = freq_aug
        assert(self.freq_aug in ['none', None, 'v0', 'v1'])
        self.rnd_pad = rnd_pad
        self.p_freq_aug = p_freq_aug
        self.bnd_width_freq_aug = bnd_width_freq_aug
        self.other_domain_root = other_domain_root
        if self.other_domain_root is not None:
            if isinstance(self.other_domain_root, (list, tuple)):
                self.other_domain_list = []
                for i_other_domain_root in self.other_domain_root:
                    self.other_domain_list += list(filter(filter_img, glob.glob(i_other_domain_root + '/*.*')))
            else:
                self.other_domain_list = list(filter(filter_img, glob.glob(self.other_domain_root + '/*.*')))
        self.stain_aug = stain_aug
        self.stain_mix = stain_mix
        self.size_aug = size_aug

        self.hist_aug = hist_aug
        if self.hist_aug:
            self.hist_aug_extra_path = glob.glob(hist_aug_extra_path + '/*.jpg') + glob.glob(hist_aug_extra_path + '/*.png') + glob.glob(hist_aug_extra_path + '/*.bmp')

        self.with_canny = with_canny

        self.images = []
        self.segs = []

        self.return_dist = return_dist
        if self.return_dist:
            self.n_dist = n_dist
            self.inst_labels = []

        if self.stain_aug or self.stain_mix:
            self.he_list = []
            self.c_list = []

        self.getbnd_online = getbnd_online
        if not self.getbnd_online:
            self.bnds = []

        if self.with_canny:
            self.canny_images = []

        # if self.freq_aug:
        #     self.source_freqs = []

        print('Preparing Dataset: ' + self.image_root)

        qbar = tqdm.tqdm(self.name_list)
        for idx, name in enumerate(qbar):
            image = skimage.io.imread(os.path.join(self.image_root, name))[:, :, 0:3]
            self.stain_norm = stain_norm
            if self.stain_norm:
                image = normalizeStaining(image)

            if self.stain_aug or self.stain_mix:
                try:
                    curhe, curc = estimateHE(image, Io=255)
                except:
                    curhe = 0
                    curc = 0
                self.he_list.append(curhe)
                self.c_list.append(curc)

            if self.with_canny:
                canny_image = []
                for ic in range(3):
                    canny_image.append(cv2.Canny(image[:,:,ic], 100, 200))
                canny_image = np.stack(canny_image, axis=2)
                self.canny_images.append(canny_image)

            image = np.float32(image) / 255.0
            self.images.append(image)

            msk_name = os.path.join(self.label_root, name)
            if not os.path.exists(msk_name):
                msk_name = msk_name[:-4] + '.bmp'
            if not os.path.exists(msk_name):
                msk_name = msk_name[:-4] + '.tif'
            if not os.path.exists(msk_name):
                msk_name = msk_name[:-4] + '.png'
            if not os.path.exists(msk_name):
                msk_name = msk_name[:-4] + '.mat'
            if not os.path.exists(msk_name):
                msk_name = msk_name[:-4] + '.npy'
            if not os.path.exists(msk_name):
                print(name, msk_name)
                raise(BaseException)
            if msk_name.endswith('.bmp') or msk_name.endswith('.png') or msk_name.endswith('.tif'):
                label = skimage.io.imread(msk_name).astype("int16")
            elif msk_name.endswith('.mat'):
                labelmat = scipy.io.loadmat(msk_name)
                label = (labelmat['inst_map']).astype("int16")
                # clslabel = (labelmat['inst_map']).astype("int16")
            else:
                label = np.load(msk_name).astype("int16")



            if len(label.shape) == 3:
                label = label[:, :, 0]

            if self.return_dist:
                self.inst_labels.append(label.copy())


            if not self.getbnd_online:
                try:
                    os.makedirs(os.path.join(self.image_root, 'cache_segbnd'), exist_ok=True)
                    label = np.load(os.path.join(self.image_root, 'cache_segbnd', os.path.basename(msk_name[0:-4])+'.npy'))
                    qbar.set_description('dataset: {}, sample: {}/{}, n_obj: {}/{}'.format(self.image_root, idx+1, len(self.name_list), len(np.unique(label[label>0])), len(np.unique(label[label>0]))))
                except:
                    segmsk = np.uint8(label>0)
                    bndmsk = np.zeros(label.shape, dtype=np.uint8)
                    if segmsk.any():
                        cset = np.unique(label[label>0])
                        for ic in cset:
                            icmap = label==ic
                            bndmsk[GetBoundary(icmap, width=1, merge='sum') > 0] = 1
                            qbar.set_description('dataset: {}, sample: {}/{}, n_obj: {}/{}'.format(self.image_root, idx+1, len(self.name_list), ic+1, len(np.unique(label[label>0]))))
                    label = np.stack((segmsk, bndmsk), axis=2)
                    np.save(os.path.join(self.image_root, 'cache_segbnd', os.path.basename(msk_name[0:-4])+'.npy'), label)

            self.segs.append(label)


            
        if domain_confidence is not None:
            assert(len(domain_confidence) == len(self.images))
            self.domain_confidence = domain_confidence
        else:
            self.domain_confidence = [1.0 for tmpi in range(len(self.images))]
        self.domain_label = [domain_label for tmpi in range(len(self.images))]

        self.max_rnd_resz = max_rnd_resz


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        image = self.images[idx].copy()
        seg = self.segs[idx].copy()
        if self.with_canny:
            canny_image = self.canny_images[idx].copy()

        h, w, _ = image.shape

        if self.return_dist:
            inst_label = self.inst_labels[idx].copy()


        if self.if_training and self.aug:
            if self.gamma_aug:
                gamma_select = random.random()
                if gamma_select >= 0.667:
                    rnd_gamma = 1.0 + random.random() * 5 # 1 - 5
                    image = exposure.adjust_gamma(image, gamma=rnd_gamma)
                elif gamma_select >= 0.333:
                    rnd_gamma = 1.0 - 0.75 * random.random() # 0.25 - 1
                    image = exposure.adjust_gamma(image, gamma=rnd_gamma)

        if self.if_training and self.aug and self.rnd_pad:
            if random.random() >= 0.5:
                h, w, _ = image.shape
                ph = random.randint(1, min(20, h//20))
                pw = random.randint(1, min(20, h//20))
                image = np.pad(image, ((ph//2, ph-ph//2), (pw//2, pw-pw//2), (0, 0)), 'constant', constant_values=1.0)
                seg = np.pad(seg, ((ph//2, ph-ph//2), (pw//2, pw-pw//2), (0, 0)), 'constant', constant_values=0.0)
                if self.return_dist:
                    inst_label = np.pad(inst_label, ((ph//2, ph-ph//2), (pw//2, pw-pw//2)), 'constant', constant_values=0)

        if self.crop is not None:
            h, w, _ = image.shape
            if self.if_training and self.aug:
                if h>self.crop and w>self.crop:
                    sh = random.randint(0, h-self.crop)
                    sw = random.randint(0, w-self.crop)
                    image = image[sh:sh+self.crop, sw:sw+self.crop, :]
                    seg = seg[sh:sh+self.crop, sw:sw+self.crop, :]
                    if self.with_canny:
                        canny_image = canny_image[sh:sh+self.crop, sw:sw+self.crop, :]
                    if self.return_dist:
                        inst_label = inst_label[sh:sh+self.crop, sw:sw+self.crop]
            else:
                raiseNotImplementedError('Cropping is only used in the training stage.')

        if self.resz is not None:
            
            if self.if_training and self.aug:
                resz = (int(self.resz[0]*(random.random()*self.max_rnd_resz + 0.5)), int(self.resz[1]*(random.random()*self.max_rnd_resz + 0.5)))
            else:
                resz = self.resz
            image = cv2.resize(image, resz, interpolation=cv2.INTER_NEAREST)
            seg = cv2.resize(seg, resz, interpolation=cv2.INTER_NEAREST)
            if self.with_canny:
                canny_image = cv2.resize(canny_image, resz, interpolation=cv2.INTER_NEAREST)
            if self.return_dist:
                inst_label = cv2.resize(inst_label, resz, interpolation=cv2.INTER_NEAREST)


            h, w, _ = image.shape
            if h<self.resz[0] or w<self.resz[1]:
                image = np.pad(image, ((0, max(0, self.resz[0]-h)), (0, max(0, self.resz[1]-w)), (0, 0)), 'edge')
                seg = np.pad(seg, ((0, max(0, self.resz[0]-h)), (0, max(0, self.resz[1]-w)), (0, 0)), 'edge')
                if self.with_canny:
                    canny_image = np.pad(canny_image, ((0, max(0, self.resz[0]-h)), (0, max(0, self.resz[1]-w)), (0, 0)), 'edge')
                if self.return_dist:
                    inst_label = np.pad(inst_label, ((0, max(0, self.resz[0]-h)), (0, max(0, self.resz[1]-w)), ), 'edge')

            h, w, _ = image.shape
            if h>self.resz[0] or w>self.resz[1]:
                sh = random.randint(0, max(0, h-self.resz[0]))
                sw = random.randint(0, max(0, w-self.resz[1]))
                image = image[sh:sh+self.resz[0], sw:sw+self.resz[1], :]
                seg = seg[sh:sh+self.resz[0], sw:sw+self.resz[1], :]
                if self.with_canny:
                    canny_image = canny_image[sh:sh+self.resz[0], sw:sw+self.resz[1], :]
                if self.return_dist:
                    inst_label = inst_label[sh:sh+self.resz[0], sw:sw+self.resz[1]]


        if self.division>1:
            h, w, _ = image.shape
            if h%self.division!=0 or w%self.division!=0:
                dh = (h//self.division+1) * self.division - h
                dw = (w//self.division+1) * self.division - w
                image = np.pad(image, ((0, dh), (0, dw), (0, 0)), 'edge')
                seg = np.pad(seg, ((0, dh), (0, dw), (0, 0)), 'edge')
                if self.with_canny:
                    canny_image = np.pad(canny_image, ((0, dh), (0, dw), (0, 0)), 'edge')
                if self.return_dist:
                    inst_label = np.pad(inst_label, ((0, dh), (0, dw)), 'edge')
            else:
                dh = 0
                dw = 0
            h, w, _ = image.shape

        if self.size_aug:
            # Not Implement
            assert(not self.return_dist)
            p_select = random.random()
            if p_select>=0.667:

                itmp_image = self.images[random.randint(0, len(self.name_list)-1)].copy()
                itmp_seg = self.segs[random.randint(0, len(self.name_list)-1)].copy()
                itmp_image = cv2.resize(itmp_image, (w,h), interpolation=cv2.INTER_NEAREST)
                itmp_seg = cv2.resize(itmp_seg, (w,h), interpolation=cv2.INTER_NEAREST)
                image = np.concatenate((image, itmp_image), axis=1)
                seg = np.concatenate((seg, itmp_seg), axis=1)

                itmp_image_2 = self.images[random.randint(0, len(self.name_list)-1)].copy()
                itmp_seg_2 = self.segs[random.randint(0, len(self.name_list)-1)].copy()
                itmp_image_2 = cv2.resize(itmp_image_2, (w,h), interpolation=cv2.INTER_NEAREST)
                itmp_seg_2 = cv2.resize(itmp_seg_2, (w,h), interpolation=cv2.INTER_NEAREST)
                itmp_image_3 = self.images[random.randint(0, len(self.name_list)-1)].copy()
                itmp_seg_3 = self.segs[random.randint(0, len(self.name_list)-1)].copy()
                itmp_image_3 = cv2.resize(itmp_image_3, (w,h), interpolation=cv2.INTER_NEAREST)
                itmp_seg_3 = cv2.resize(itmp_seg_3, (w,h), interpolation=cv2.INTER_NEAREST)
                itmp_image_2 = np.concatenate((itmp_image_2, itmp_image_3), axis=1)
                itmp_seg_2 = np.concatenate((itmp_seg_2, itmp_seg_3), axis=1)

                image = np.concatenate((image, itmp_image_2), axis=0)
                seg = np.concatenate((seg, itmp_seg_2), axis=0)
                image = cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
                seg = cv2.resize(seg, (w,h), interpolation=cv2.INTER_NEAREST)

            elif p_select>=0.333:
                sh = random.randint(0, h//4)
                sw = random.randint(0, w//4)
                eh = random.randint(3*h//4, h-1)
                ew = random.randint(3*h//4, w-1)
                image = image[sh:eh, sw:ew, :]
                seg = seg[sh:eh, sw:ew, :]
                image = cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
                seg = cv2.resize(seg, (w,h), interpolation=cv2.INTER_NEAREST)


        if self.aug and self.hist_aug:
            if random.random()>=0.5:
                rnd_select = random.randint(0, len(self.hist_aug_extra_path)-1)
                aug_image = skimage.io.imread(self.hist_aug_extra_path[rnd_select])
                if len(aug_image.shape) == 2:
                    aug_image = skimage.color.gray2rgb(aug_image)
                aug_image = cv2.resize(aug_image[:,:,0:3], dsize=(w,h)).astype(image.dtype)/255.0
                if random.random() >= 0.5:
                    ref_ = np.reshape(image, (-1, 3))
                else:
                    ref_ = -np.reshape(image, (-1, 3)) # inverse
                ref_ = np.argsort(np.argsort(ref_, axis=0), axis=0)
                values_ = np.sort(np.reshape(aug_image, (-1, 3)), axis=0)
                image = np.reshape(np.take_along_axis(values_, ref_, axis=0), (h,w,3)).astype(np.uint8)


        # Stain
        if self.if_training and self.aug and (self.stain_aug or self.stain_mix):
            if (image>0).any():
                if random.random() >= 0.5:
                    try:
                        rndstate = np.random.RandomState(random.randint(0, 12345)*(idx+12345))
                        rndidx = random.randint(0, len(self.name_list)-1)
                        rndhe, rndc = self.he_list[rndidx], self.c_list[rndidx]

                        if self.stain_aug and self.stain_mix:
                            p_choice = random.random()
                        elif self.stain_aug:
                            p_choice = 1.0
                        else: # stain_mix
                            p_choice = 0.0
                        if p_choice < 0.5:
                            r = random.random()
                            curhe, curc = self.he_list[idx], self.c_list[idx]
                            cur_HERef = rndhe*r + curhe*(1-r)
                            cur_maxCRef = rndc*r + curc*(1-r)
                        else:
                            cur_HERef = np.clip(rndstate.normal(loc=rndhe, scale=0.01), a_min=0.05, a_max=1.0)
                            cur_maxCRef = np.clip(rndstate.normal(loc=rndc, scale=0.01), a_min=0.5, a_max=5.0)
                        image = normalizeStaining(np.clip(image*255, a_min=0, a_max=255), Io=255, HERef=cur_HERef, maxCRef=cur_maxCRef)
                        image = np.float32(image) / 255.0
                    except:
                        pass


        # Freq
        if self.if_training and self.aug and self.freq_aug == 'v0':

            if random.random()>=0.5:
                p = random.random()
                if self.other_domain_root is not None:
                    aug_img_path = self.other_domain_list[random.randint(0, len(self.other_domain_list)-1)]
                    aug_img = np.float32(skimage.io.imread(aug_img_path)[:, :, 0:3]) / 255.0
                else:
                    aug_img = self.images[random.randint(0, len(self.images)-1)].copy()
                aug_img = cv2.resize(aug_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) 
                image = freq_swap_aug(image, aug_img, L=self.bnd_width_freq_aug, ratio=p)

        if self.if_training and self.aug and self.freq_aug == 'v1':

            if random.random()>=0.5:
                p = random.random()
                if self.other_domain_root is not None:
                    aug_img_path = self.other_domain_list[random.randint(0, len(self.other_domain_list)-1)]
                    aug_img = np.float32(skimage.io.imread(aug_img_path)[:, :, 0:3]) / 255.0
                else:
                    aug_img = self.images[random.randint(0, len(self.images)-1)].copy()
                aug_img = cv2.resize(aug_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                HED_img = rgb2hed(image)
                HED_aug_img = rgb2hed(aug_img)
                rnd_ic = random.randint(0, 2)
                HED_img[:, :, rnd_ic] = freq_swap_aug(HED_img[:, :, rnd_ic], HED_aug_img[:, :, rnd_ic], L=self.bnd_width_freq_aug, ratio=p)
                image = hed2rgb(HED_img)


        if self.return_he:
            tmp_image = np.zeros(image.shape, dtype=np.float32)
            for ic in range(3):
                tmp_image[:,:,ic] = (image[:,:,ic]-image[:,:,ic].min()) / (image[:,:,ic].max()-image[:,:,ic].min())
            HE, maxC = estimateHE(np.uint8(255*tmp_image))


        image = (image - norm_mean) / norm_std
        image = np.transpose(image, [2, 0, 1]).astype(np.float32)
        if self.with_canny:
            canny_image = (canny_image - canny_image.mean()) / np.clip(canny_image.std(), a_min=1e-3, a_max=np.inf)
            canny_image = np.transpose(canny_image, [2, 0, 1]).astype(np.float32)
            image = np.concatenate((image, canny_image), axis=0)

        if self.getbnd_online:
            segmsk = np.float32(seg>0)
            bndmsk = np.zeros(seg.shape, dtype=np.float32)
            if segmsk.any():
                cset = np.unique(seg[seg>0])
                for ic in cset:
                    bndmsk += GetBoundary(seg==ic, width=1)
                bndmsk[bndmsk>0] = 1.0
            segmsk = np.stack((segmsk, bndmsk), axis=0)
        else:
            segmsk = np.float32(np.transpose(seg, [2, 0, 1]))

        if self.return_dist:
            dist = np.transpose(star_dist(inst_label, self.n_dist).astype(np.float32), (2,0,1))


        outputs = [image, segmsk]

        if self.return_dist:
            outputs += [dist, ]

        if self.return_he:
            # print(HE, HERef)
            # print(maxC, maxCRef)
            outputs += [HE, maxC, ]

        if self.return_contour:
            contours = GetBoundary(segmsk, width=5, class_axis=0)
            outputs += [contours, ]

        if self.return_domain_lbl:
            domain_label = np.array([self.domain_label[idx], ], dtype=np.int64)
            outputs += [domain_label, ]

        if self.return_domain_confidence:
            domain_confidence = np.array([self.domain_confidence[idx], ], dtype=np.float32)
            outputs += [domain_confidence, ]
            
        if self.return_tgt_image:
            outputs += [image, ]
        
        return outputs



##########################################################################################################################################
##########################################################################################################################################

def getTrainDataLoaders(image_root, label_root, batch_size, name_list=None, \
        train_crop=None, test_crop=None, resz=None, domain_label=0, domain_confidence=None, domain_balanced=False, num_samples=None, shuffle=True, aug=True, \
        return_domain_lbl=False, return_domain_confidence=False, return_tgt_image=False, return_contour=False, return_he=False, \
        gamma_aug=True, stain_aug=False, stain_mix=False, stain_norm=False, size_aug=False, with_canny=False, freq_aug=None, rnd_pad=False, use_prefetch=False, max_rnd_resz=1.0, **kwargs):

    if isinstance(resz, int):
        resz = (resz, resz)

    assert(len(name_list)==2)

    for k, v in kwargs.items():
        print(k, v)

    list_all = np.arange(0, len(image_root))

    if isinstance(image_root, (list, tuple)):
        trainset = []
        for i_domain, (i_image_root, i_label_root, i_name_list_0) in enumerate(zip(image_root, label_root, name_list[0])):
            other_domain_root = [image_root[i_other] for i_other in np.setdiff1d(list_all, i_domain).tolist()]
            trainset.append(PathologyDataset(\
                i_image_root, i_label_root, name_list=i_name_list_0, if_training=True, crop=train_crop, resz=resz, \
                domain_label=i_domain, domain_confidence=domain_confidence, \
                return_domain_lbl=return_domain_lbl, return_domain_confidence=return_domain_confidence, return_tgt_image=return_tgt_image, return_contour=return_contour, return_he=return_he,\
                aug=aug, gamma_aug=gamma_aug, stain_aug=stain_aug, stain_mix=stain_mix, stain_norm=stain_norm, size_aug=size_aug, with_canny=with_canny, freq_aug=freq_aug, rnd_pad=rnd_pad, other_domain_root=other_domain_root, max_rnd_resz=max_rnd_resz, **kwargs))
        trainset = ConcatDataset(trainset)
    else:
        trainset = PathologyDataset(image_root, label_root, name_list=name_list[0], if_training=True, crop=train_crop, resz=resz, \
            domain_label=domain_label, domain_confidence=domain_confidence, \
            return_domain_lbl=return_domain_lbl, return_domain_confidence=return_domain_confidence, return_tgt_image=return_tgt_image, return_contour=return_contour, return_he=return_he,\
            aug=aug, gamma_aug=gamma_aug, stain_aug=stain_aug, stain_mix=stain_mix, stain_norm=stain_norm, size_aug=size_aug, with_canny=with_canny, freq_aug=freq_aug, max_rnd_resz=max_rnd_resz, **kwargs)
    
    if domain_balanced:

        domain_labels = []
        for iset in trainset.datasets:
            domain_labels += iset.domain_label
        domain_labels = np.array(domain_labels)
        domain_lable_types = np.unique(domain_labels)
        sampling_weights = np.zeros(domain_labels.shape, dtype=np.float32)
        for i_d_type in domain_lable_types:
            sampling_weights[domain_labels == i_d_type] = 1.0 / np.mean(domain_labels == i_d_type)
        shuffle = False
        if num_samples is None:
            num_samples = len(trainset)
        elif num_samples < len(trainset):
            print('Warning: sampling number is smaller than the length of training dataset.')

        sampler = WeightedRandomSampler(sampling_weights, num_samples=num_samples, replacement=True)
        
    else:
        shuffle = shuffle
        sampler = None

    # min(max(batch_size, 2), 10)
    if use_prefetch:
        trainloader = DataLoaderX(trainset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=1 , drop_last=True)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=1, drop_last=True)

    return trainloader


##########################################################################################################################################
##########################################################################################################################################

def getSeparateTrainDataLoaders(image_root, label_root, batch_size, name_list=None, \
        train_crop=None, test_crop=None, resz=None, stain_norm=False, domain_label=0, domain_confidence=None, num_samples=None, aug=True, \
        return_domain_lbl=False, return_domain_confidence=False, return_tgt_image=False, \
        gamma_aug=True, with_canny=False):

    # Always domain balanced
    if isinstance(resz, int):
        resz = (resz, resz)

    assert(len(name_list)==2)
    assert(isinstance(image_root, (list, tuple)))

    trainloader = []
    for i_domain, (i_image_root, i_label_root, i_name_list_0) in enumerate(zip(image_root, label_root, name_list[0])):
        trainset = FundusDataset(\
            i_image_root, i_label_root, name_list=i_name_list_0, if_training=True, crop=train_crop, resz=resz, stain_norm=stain_norm,\
            domain_label=i_domain, domain_confidence=domain_confidence, \
            return_domain_lbl=return_domain_lbl, return_domain_confidence=return_domain_confidence, return_tgt_image=return_tgt_image, \
            aug=aug, gamma_aug=gamma_aug, with_canny=with_canny)
        sampler = RandomSampler(trainset, num_samples=num_samples, replacement=True)
        trainloader.append(DataLoader(trainset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=min(max(batch_size, 1), 2), drop_last=True))
    return trainloader


##########################################################################################################################################
##########################################################################################################################################

class PathologyTestDataset(Dataset):
    def __init__(self, image_root, label_root, name_list=None, crop=None, resz=None, division=1, \
                domain_label=0, domain_confidence=None, getbnd_preload=False, getbnd_online=False, stain_norm=False, \
                return_domain_lbl=False, return_domain_confidence=False, return_tgt_image=False, imagenet_norm=True):

        self.image_root = image_root
        self.label_root = label_root
        if name_list is not None:
            self.name_list = name_list
        else:
            self.name_list = os.listdir(self.image_root)
            self.name_list.sort()


        self.resz = resz
        self.crop = crop
        self.division = division

        self.return_domain_lbl = return_domain_lbl
        self.return_domain_confidence = return_domain_confidence
        self.return_tgt_image = return_tgt_image


        if domain_confidence is not None:
            assert(len(domain_confidence) == len(self.name_list))
            self.domain_confidence = domain_confidence
        else:
            self.domain_confidence = [1.0 for tmpi in range(len(self.name_list))]
        self.domain_label = [domain_label for tmpi in range(len(self.name_list))]

        self.getbnd_preload = getbnd_preload

        if self.getbnd_preload:
            os.makedirs(os.path.join(self.image_root, 'cache_segbnd'), exist_ok=True)
            self.segs = []
            qbar = tqdm.tqdm(enumerate(self.name_list))
            for idx, name in qbar:

                msk_name = os.path.join(self.label_root, name)
                if not os.path.exists(msk_name):
                    msk_name = msk_name[:-4] + '.bmp'
                if not os.path.exists(msk_name):
                    msk_name = msk_name[:-4] + '.png'
                if not os.path.exists(msk_name):
                    msk_name = msk_name[:-4] + '.tif'
                if not os.path.exists(msk_name):
                    msk_name = msk_name[:-4] + '.mat'
                if not os.path.exists(msk_name):
                    msk_name = msk_name[:-4] + '.npy'

                if not os.path.exists(msk_name):
                    print(msk_name)
                    raise(BaseException)
                    
                try:
                    seg = np.load(os.path.join(self.image_root, 'cache_segbnd', os.path.basename(msk_name)[0:-4]+'.py.npy'))
                    qbar.set_description('dataset: {}, sample: {}/{}'.format(self.image_root, idx+1, len(self.name_list)))
                except:

                    if msk_name.endswith('.bmp') or msk_name.endswith('.png') or msk_name.endswith('.tif'):
                        label = skimage.io.imread(msk_name).astype("int16")
                    elif msk_name.endswith('.mat'):
                        labelmat = scipy.io.loadmat(msk_name)
                        label = (labelmat['inst_map']).astype("int16")
                    else:
                        label = np.load(msk_name).astype("int16")
                    if len(label.shape) == 3:
                        label = label[:, :, 0]
                    segmsk = np.float32(label>0)
                    bndmsk = np.zeros(label.shape, dtype=np.float32)
                    if segmsk.any():
                        cset = np.unique(label[label>0])
                        for ic in cset:
                            bndmsk += GetBoundary(label==ic, width=1)
                        bndmsk[bndmsk>0] = 1.0
                    seg = np.stack((segmsk, bndmsk), axis=0)
                    np.save(os.path.join(self.image_root, 'cache_segbnd', os.path.basename(msk_name)[0:-4]+'.py.npy'), seg)
                    qbar.set_description('dataset: {}, sample: {}/{}, n_obj: {}'.format(self.image_root, idx+1, len(self.name_list), label.max()))

                self.segs.append(seg)


        self.getbnd_online = getbnd_online

        self.stain_norm = stain_norm
        self.imagenet_norm = imagenet_norm

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):


        ori_img = skimage.io.imread(os.path.join(self.image_root, self.name_list[idx]))[:, :, 0:3]
        if self.stain_norm:
            ori_img = normalizeStaining(ori_img)

        image = np.float32(ori_img.copy()) / 255.0

        msk_name = os.path.join(self.label_root, self.name_list[idx])
        if not os.path.exists(msk_name):
            msk_name = msk_name[:-4] + '.bmp'
        if not os.path.exists(msk_name):
            msk_name = msk_name[:-4] + '.png'
        if not os.path.exists(msk_name):
            msk_name = msk_name[:-4] + '.tif'
        if not os.path.exists(msk_name):
            msk_name = msk_name[:-4] + '.mat'
        if not os.path.exists(msk_name):
            msk_name = msk_name[:-4] + '.npy'
        if not os.path.exists(msk_name):
            print(msk_name)
            raise(BaseException)
        if msk_name.endswith('.bmp') or msk_name.endswith('.png') or msk_name.endswith('.tif'):
            seg = skimage.io.imread(msk_name).astype("int16")
        elif msk_name.endswith('.mat'):
            labelmat = scipy.io.loadmat(msk_name)
            seg = (labelmat['inst_map']).astype("int16")
        else:
            seg = np.load(msk_name).astype("int16")


        if len(seg.shape) == 3:
            seg = seg[:, :, 0]

        # seg = skimage.io.imread(msk_name)
        # if len(seg.shape) == 3:
        #     seg = seg[:, :, 0]
        # seg[seg <= 50] = 2
        # seg[np.logical_and(seg>50, seg<=200)] = 1
        # seg[seg >= 255] = 0

        ori_seg = seg.copy()

        h, w, _ = image.shape

        if self.getbnd_preload:
            segbnd = self.segs[idx]
        else:
            segbnd = None

        if self.crop is not None: # ONLY used for test-time adaption OR debugging
            sh = random.randint(0, h-self.crop)
            sw = random.randint(0, w-self.crop)
            image = image[sh:sh+self.crop, sw:sw+self.crop, :]
            seg = seg[sh:sh+self.crop, sw:sw+self.crop]
            if segbnd is not None:
                segbnd = segbnd[:, sh:sh+self.crop, sw:sw+self.crop]


        if self.resz is not None:
            resz = self.resz
            image = cv2.resize(image, resz, interpolation=cv2.INTER_NEAREST)
            seg = cv2.resize(seg, resz, interpolation=cv2.INTER_NEAREST)
            if segbnd is not None:
                segbnd = np.transpose(segbnd, (1,2,0))
                segbnd = cv2.resize(segbnd, resz, interpolation=cv2.INTER_NEAREST)
                segbnd = np.transpose(segbnd, (2,0,1))

        h, w, _ = image.shape
        
        dh = 0
        dw = 0
        if self.division>1:
            h, w, _ = image.shape
            if h%self.division!=0 or w%self.division!=0:
                dh = (h//self.division+1) * self.division - h
                dw = (w//self.division+1) * self.division - w
                image = np.pad(image, ((0, dh), (0, dw), (0, 0)), 'edge')
                seg = np.pad(seg, ((0, dh), (0, dw)), 'edge')
                if segbnd:
                    segbnd = np.pad(segbnd, ((0, 0), (0, dh), (0, dw)), 'edge')


            h, w, _ = image.shape

        if self.imagenet_norm:
            image = (image - norm_mean) / norm_std
            image = np.transpose(image, [2, 0, 1]).astype(np.float32)


        if not self.getbnd_preload and self.getbnd_online:
            segmsk = np.float32(seg>0)
            bndmsk = np.zeros(seg.shape, dtype=np.float32)
            if segmsk.any():
                cset = np.unique(seg[seg>0])
                for ic in cset:
                    bndmsk += GetBoundary(seg==ic, width=1)
                bndmsk[bndmsk>0] = 1.0
            seg = np.stack((segmsk, bndmsk), axis=0)

        elif self.getbnd_preload:
            seg = segbnd



        outputs = [image, seg, np.array([dh, dw]), ori_seg, ori_img]

        if self.return_domain_lbl:
            domain_label = np.array([self.domain_label[idx], ], dtype=np.int64)
            outputs += [domain_label, ]

        if self.return_domain_confidence:
            domain_confidence = np.array([self.domain_confidence[idx], ], dtype=np.float32)
            outputs += [domain_label, ]
            
        if self.return_tgt_image:
            outputs += [image, ]
        
        return outputs



def collate_return_list(batch):
    n_data = len(batch)
    n_out = len(batch[0])
    batch = [ [batch[i_data][i_out] for i_data in range(n_data)] for i_out in range(n_out) ]
    return batch

def getTestDataLoaders(image_root, label_root, batch_size, name_list=None, \
        train_crop=None, test_crop=None, resz=None, resz_tgt=False, stain_norm=False, domain_label=0, domain_confidence=None, \
        return_domain_lbl=False, return_domain_confidence=False, return_tgt_image=False, getbnd_preload=False, getbnd_online=False, num_workers=2, shuffle=False, **kwargs):

    if isinstance(resz, int):
        resz = (resz, resz)

    if isinstance(image_root, (list, tuple)):
        testset = []
        for i_domain, (i_image_root, i_label_root, i_name_list) in enumerate(zip(image_root, label_root, name_list)):
            testset.append(PathologyTestDataset(i_image_root, i_label_root, name_list=i_name_list, crop=test_crop, resz=resz, stain_norm=stain_norm, \
                                domain_label=i_domain, domain_confidence=domain_confidence, \
                                return_domain_lbl=return_domain_lbl, return_domain_confidence=return_domain_confidence, return_tgt_image=return_tgt_image, getbnd_preload=getbnd_preload, getbnd_online=getbnd_online, **kwargs))
        testset = ConcatDataset(testset)
    else:
        testset = PathologyTestDataset(image_root, label_root, name_list=name_list, crop=test_crop, resz=resz, stain_norm=stain_norm, \
            domain_label=domain_label, domain_confidence=domain_confidence, \
            return_domain_lbl=return_domain_lbl, return_domain_confidence=return_domain_confidence, return_tgt_image=return_tgt_image, getbnd_preload=getbnd_preload, getbnd_online=getbnd_online, **kwargs)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, sampler=None, num_workers=num_workers, collate_fn=collate_return_list)

    return testloader

