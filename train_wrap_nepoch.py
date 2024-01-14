import torch
import time
import sys
from load_save_model import checkpoint_save_stage
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm
import math
import numpy as np
import skimage
import skimage.io
from dataset.pathology.normalizeStaining import normalizeStaining, estimateHE
import random

norm_mean=np.array([[[0.485, 0.456, 0.406]]])
norm_std=np.array([[[0.229, 0.224, 0.225]]])

class Trainer():

    def __init__(self, loss=None, metric=None, log_dir=None, validate_every=2, validate_every_end=None, \
                    verborrea=True, data_label_split_idx=1, USE_CUDA=True, save_every=10, valid_in_train=False, \
                    randconv=None, with_canny=False, visualize=False, visualize_dir=None, restain=False):
        self.loss_func = loss
        self.metric = metric
        self.verborrea = verborrea
        self.USE_CUDA = torch.cuda.is_available() and USE_CUDA

        self.validate_every = validate_every
        if validate_every_end is None:
            self.validate_every_end = validate_every
        else:
            self.validate_every_end = validate_every_end
        self.log_dir = log_dir

        self.data_label_split_idx = data_label_split_idx
        self.save_every = save_every

        self.valid_in_train = valid_in_train

        self.randconv = randconv
        self.with_canny = with_canny

        self.visualize = visualize
        self.visualize_dir = visualize_dir

        self.restain = restain
            


    def Train(self, model, optimizer, TrainSet, TestSet, Train_mode, Model_name, Dataset, epochs=None, scheduler=None, fix_amp_epoch=None):

        if self.loss_func is None:
            print("Loss function not set,exiting...")
            sys.exit()

        if scheduler is not ReduceLROnPlateau:
            assert(epochs is not None and epochs > 0)
            if scheduler is None:
                scheduler = StepLR(optimizer, step_size=epochs//5, gamma=0.2)
        else:
            assert(self.valid_in_train)

        path_checkpoint = os.getcwd()+'/CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset+'/CHECKPOINT.t7'
        print('Checkpoint path',path_checkpoint)
        
        if self.visualize:
            if self.visualize_dir is None:
                self.visualize_dir = os.path.join(os.getcwd(), 'vis_train_tmp', Model_name+'_'+Train_mode+'_'+Dataset)
                os.makedirs(self.visualize_dir, exist_ok=True)

        max_lr, list_lr = self.update_list_lr(optimizer)
        trainloss_to_fil=[]
        testloss_to_fil=[]
        trainMetric_to_fil=[]
        testMetric_to_fil=[]
    
        if isinstance(scheduler,ReduceLROnPlateau):
            patience_num=scheduler.patience

        else:
            print('Scheduler not supported. But training will continue if epochs are specified.')
            if epochs==None:
                print('WARNING!!!! Number of epochs not specified')
                sys.exit()
            patience_num='nothing'
            
        parameters=[[],[],patience_num,optimizer.param_groups[0]['weight_decay']]#first list for epochs, second for learning rate,3rd patience, 4th weight_decay,5 for time
        parameters[1].append(list_lr)
        
        epoch = 0

        keep_training = True
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler_mode = scheduler.mode
            if scheduler_mode == 'min':
                scheduler.step(torch.FloatTensor([math.inf]))
            else:
                scheduler.step(torch.FloatTensor([-math.inf]))

        else:
            best_test = math.inf
            # scheduler.step()

        since_init = time.time()

        while keep_training:

            epoch = epoch+1
    
            if self.verborrea:
                print('-' * 20)

            if isinstance(scheduler, ReduceLROnPlateau):
                patience = scheduler.patience
            else:
                patience = -1

            max_lr, list_lr = self.update_list_lr(optimizer)
            if epochs !=None:
                if self.verborrea: 
                    print('Epoch {}/{},  lr={}. patience={}, weight decay={}'.format(epoch, epochs,max_lr,patience,optimizer.param_groups[0]['weight_decay']))
            else:
                if self.verborrea:
                    print('Epoch {}, lr={}, patience={}, weight decay={}'.format(epoch,max_lr,patience,optimizer.param_groups[0]['weight_decay']))
            with open(self.log_dir, 'a') as f:
                f.write('-'*30+'\n')
                f.write('Epoch {}, lr={}, patience={}, weight decay={}'.format(epoch,max_lr,patience,optimizer.param_groups[0]['weight_decay'])+'\n')


            if self.verborrea:
                print ('TRAIN STATISTICS')
            model.train()


            if fix_amp_epoch is not None and epoch > fix_amp_epoch:
                if model.amp_norm is not None:
                    model.amp_norm.fix_amp = True

            train_loss,train_metric= self.train_scratch(model, TrainSet, optimizer, epoch) #Training happens here!
            
            if self.valid_in_train:
                if epoch % self.validate_every == 0 :
                    if self.verborrea:
                        print ('TEST STATISTICS')
                    print('Validating at epoch',epoch)
                    model.eval()
                    test_loss, test_metric= self.evaluate(model,TestSet,epoch)

                    trainloss_to_fil.append(train_loss)
                    testloss_to_fil.append(test_loss)
                    trainMetric_to_fil.append(train_metric)
                    testMetric_to_fil.append(test_metric)

                    if isinstance(scheduler, ReduceLROnPlateau):
                        prev_num_bad_epochs=scheduler.num_bad_epochs
                        if self.verborrea:
                            print('-----' * 10)
                        if scheduler_mode =='min':
                            save=(test_loss< scheduler.best)
                            scheduler.step(test_loss)
                        else:
                            save=(test_metric>scheduler.best)
                            scheduler.step(test_metric)
                            print('Best', scheduler.best)

                        if save:
                            checkpoint_save_stage(model,trainloss_to_fil,testloss_to_fil,trainMetric_to_fil,testMetric_to_fil,parameters,Model_name,Train_mode,Dataset)
                            check_load=0
                        if scheduler.num_bad_epochs==0 and prev_num_bad_epochs==scheduler.patience and not save:
                            max_lr,list_lr=self.update_list_lr(optimizer)
                            parameters[0].append(epoch)
                            parameters[1].append(max_lr)
                            model.load_state_dict(torch.load(path_checkpoint))
                            check_load=check_load+1
                            if self.verborrea: print ('Checkpoint loaded')
                            self.validate_every = max(self.validate_every_end, self.validate_every//2) # Change Validation Frequency

                        if max_lr<=scheduler.eps or check_load>=8:
                            keep_training=False
                    else:
                        scheduler.step()
                        if test_loss<=best_test:
                            checkpoint_save_stage(model, trainloss_to_fil, testloss_to_fil, trainMetric_to_fil, testMetric_to_fil, parameters, Model_name, Train_mode, Dataset)
                            print ('Epoch ' + str(epoch) + ': Checkpoint Saved')

                if epoch%self.save_every==0 or epoch == epochs:
                    torch.save(model.state_dict(), os.getcwd()+'/CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset+'/CHECKPOINT_'+'epoch_'+str(epoch)+'.t7')


            elif epoch%self.save_every==0 or epoch == epochs:
                scheduler.step()
                checkpoint_save_stage(model, trainloss_to_fil, testloss_to_fil, trainMetric_to_fil, testMetric_to_fil, parameters, Model_name, Train_mode, Dataset, checkpoint_suffix='_epoch_'+str(epoch))
                print ('Epoch ' + str(epoch) + ': Checkpoint Saved')

            else:
                scheduler.step()

            if epochs!=None:
                if epoch>=epochs:
                    keep_training=False


        if epochs!=0 and isinstance(scheduler, ReduceLROnPlateau):
            model.load_state_dict(torch.load(path_checkpoint))
            if self.verborrea:
                print ('Checkpoint loaded')
    
        parameters[0].append(epoch)
        print ('FINAL TRAIN STATISTICS')
        train_loss, train_metric= self.evaluate(model,TrainSet)
        trainloss_to_fil.append(train_loss)
        trainMetric_to_fil.append(train_metric)

        if self.valid_in_train:
            print ('FINAL TEST STATISTICS')
            test_loss, test_metric= self.evaluate(model,TestSet)
            testloss_to_fil.append(test_loss)
            testMetric_to_fil.append(test_metric)
        
        time_elapsed=time.time()-since_init
        print('Total time elapsed',time_elapsed)
        parameters.append(time_elapsed)

        return (trainloss_to_fil,testloss_to_fil,trainMetric_to_fil,testMetric_to_fil,parameters)


    def train_scratch(self, model, DataSet, optimizer, epoch=None): #eval is not correct in the method
        _loss = 0.0
        _loss_terms = {}
        model.train()
        kwargs ={}
        kwargs['state'] = 'train'

        pbar = tqdm(DataSet)
        st = time.time()

        for batch_idx, data in enumerate(pbar):

            optimizer.zero_grad()

            inputs = data[0:self.data_label_split_idx]

            # if self.restain:
            #     assert(DataSet.dataset.datasets[0].return_he)
            #     HE = data[2].data.cpu().numpy()
            #     maxC = data[3].data.cpu().numpy()

            targets = data[1:]
            
            if self.USE_CUDA:
                if isinstance(inputs, (list, tuple)):
                    if len(inputs) == 1:
                        inputs = inputs[0].cuda()
                    else:
                        inputs = [input.cuda() for input in inputs]
                else:
                    inputs  = inputs.cuda()
                if not(isinstance(targets, (list, tuple))):
                    targets = [targets, ]
                targets = [target.cuda() for target in targets]

            batch_size = inputs.shape[0]

            if self.with_canny:
                
                inputs = torch.split(inputs, split_size_or_sections=3, dim=1)
                n_split = len(inputs)
                inputs = torch.cat(inputs, dim=0)
                targets = [ torch.cat([target, ]*n_split, dim=0) for target in targets ]

            if self.visualize and batch_idx%10==0:
                img = inputs[0].clone()
                # imgminv = (img.min(dim=-2, keepdim=True)[0]).min(dim=-1, keepdim=True)[0]
                # imgmaxv = (img.max(dim=-2, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]
                img = (img - img.min()) / (img.max() - img.min())
                img = np.transpose(img.data.cpu().numpy(), [1,2,0])
                img2show = [img, ]


            if self.randconv is not None:
                with torch.no_grad():
                    inputs_new = [inputs, ]
                    if isinstance(self.randconv, (list, tuple)):
                        for i_randconv in self.randconv:
                            i_randconv.randomize()
                            inputs_new.append(i_randconv(inputs))
                            # print(inputs.min().item(), inputs.max().item(), i_randconv(inputs).min().item(), i_randconv(inputs).max().item())
                    else:
                        self.randconv.randomize()
                        inputs_new.append(self.randconv(inputs))
                    if self.restain:
                        ori_shape = inputs.shape
                        sorted_rep, _ = torch.sort(inputs.view(ori_shape[0], ori_shape[1], -1), dim=-1)
                        for i_aug, aug_inputs in enumerate(inputs_new[1:]):
                            _, argsort_x_idx = torch.sort(aug_inputs.view(ori_shape[0], ori_shape[1], -1), dim=-1)
                            inverse_argsort_x_idx = argsort_x_idx.argsort(-1)
                            inputs_new[1+i_aug] = torch.gather(sorted_rep, -1, inverse_argsort_x_idx).view(ori_shape)


                    # for i_aug, aug_inputs in enumerate(inputs_new[1:]):
                    #     aug_inputs = aug_inputs.data.cpu()
                    #     minv = aug_inputs.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
                    #     maxv = aug_inputs.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
                    #     aug_inputs = (aug_inputs-minv) / (maxv-minv)
                    #     for ib in range(batch_size):
                    #         ib_aug_input = np.uint8(aug_inputs[ib].numpy()*255)
                    #         ib_aug_input = np.transpose(ib_aug_input, (1,2,0))
                    #         ib_aug_input = normalizeStaining(ib_aug_input, HERef=HE[ib], maxCRef=maxC[ib])
                    #         ib_aug_input = np.float32(ib_aug_input)/255.0
                    #         ib_aug_input = (ib_aug_input - norm_mean) / norm_std
                    #         ib_aug_input = np.transpose(ib_aug_input, (2,0,1))
                    #         inputs_new[i_aug+1][ib] = torch.FloatTensor(ib_aug_input).to(inputs_new[i_aug+1][ib].device)

                inputs = torch.cat(inputs_new, dim=0)

                if self.visualize and batch_idx%10==0:
                    img = inputs[batch_size].clone()
                    # imgminv = (img.min(dim=-2, keepdim=True)[0]).min(dim=-1, keepdim=True)[0]
                    # imgmaxv = (img.max(dim=-2, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]
                    img = (img - img.min()) / (img.max() - img.min())
                    img = np.transpose(img.data.cpu().numpy(), [1,2,0])
                    img2show.append(img)
                    img2show.append(np.abs(img2show[0] - img2show[1]))


            predictions =  model(inputs)

            if self.visualize and batch_idx%10==0:

                img = predictions[0][0].clone().sum(dim=0, keepdim=True)
                imgminv = (img.min(dim=-2, keepdim=True)[0]).min(dim=-1, keepdim=True)[0]
                imgmaxv = (img.max(dim=-2, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]
                img = (img - imgminv) / (imgmaxv - imgminv)
                img = np.transpose(np.repeat(img.data.cpu().numpy(), 3, axis=0), [1,2,0])
                img2show.append(img)

                img = targets[0][0].clone().sum(dim=0, keepdim=True)
                imgminv = (img.min(dim=-2, keepdim=True)[0]).min(dim=-1, keepdim=True)[0]
                imgmaxv = (img.max(dim=-2, keepdim=True)[0]).max(dim=-1, keepdim=True)[0]
                img = (img - imgminv) / (imgmaxv - imgminv)
                img = np.transpose(np.repeat(img.data.cpu().numpy(), 3, axis=0), [1,2,0])
                img2show.append(img)

                # for i_ in img2show:
                #     print(i_.shape)

                img2show = np.uint8(np.concatenate(img2show, axis=1) * 255)
                skimage.io.imsave(os.path.join(self.visualize_dir, 'epoch_'+str(epoch)+'_iter_'+str(batch_idx)+'.png'), img2show)


            total_loss, loss_info = self.loss_func(predictions, targets, **kwargs)
            total_loss.backward()
            optimizer.step()
            _loss += total_loss.item()

            print_info = 'Processing %{:.3f}({:d}/{:d}) || '.format(float(batch_idx)/len(pbar)*100, batch_idx, len(pbar))

            for k, v in loss_info.items():
                if not(k in _loss_terms.keys()):
                    _loss_terms.update({k: v.item()})
                else:
                    _loss_terms.update({k: v.item()+_loss_terms[k]})
                print_info += k + ': {:.5f}; '.format(v.item())
            print_info += 'Total Loss: {:.5f}'.format(total_loss.item())

            pbar.set_description(print_info)

        _loss_average = _loss/len(DataSet)
        et = time.time()

        if self.verborrea:
            print('-'*10 + 'train' + '-'*10)
            for k, v in _loss_terms.items():
                print(k+'(Sum): ',  v)
            print('Loss (Sum): ', _loss)
            print(' ')
            for k, v in _loss_terms.items():
                print(k+'(Avg): ',  v/len(DataSet))
            print('Loss (Avg): ', _loss_average)
            print(' ')
            with open(self.log_dir, 'a') as f:
                f.write('-'*10 + 'train' + '-'*10 + '\n')
                for k, v in _loss_terms.items():
                    f.write('{:s}  (Avg): {:.7f}'.format(k, v/len(DataSet))+'\n')
                f.write('Loss (Avg): {:.7f}'.format(_loss_average)+'\n')
                f.write('\n')

        print('Used Time: ', et - st)


        return _loss_average, 0.0


    def evaluate(self, model, DataSet, epoch=None):

        with torch.no_grad():
            _loss = 0.0
            _loss_terms = {}
            model.eval()
            kwargs ={}
            kwargs['state'] = 'val'
            pbar = tqdm(DataSet)
            for batch_idx, data in enumerate(pbar):

                inputs = data[0:self.data_label_split_idx]
                targets = data[1:]
                if self.USE_CUDA:
                    if isinstance(inputs, (list, tuple)):
                        if len(inputs) == 1:
                            inputs = inputs[0].cuda()
                        else:
                            inputs = [input.cuda() for input in inputs]
                    else:
                        inputs  = inputs.cuda()
                if not(isinstance(targets, (list, tuple))):
                    targets = [targets, ]
                targets = [target.cuda() for target in targets]

                if self.with_canny:
                    batch_size = inputs.shape[0]
                    inputs = torch.split(inputs, split_size_or_sections=3, dim=1)
                    n_split = len(inputs)
                    inputs = torch.cat(inputs, dim=0)
                    targets = [ torch.cat([target, ]*n_split, dim=0) for target in targets ]


                
                predictions =  model(inputs)

                total_loss, loss_info = self.loss_func(predictions, targets, **kwargs)
                _loss += total_loss.item()

                print_info = 'Processing %{:.3f}({:d}/{:d}) || '.format(float(batch_idx)/len(pbar)*100, batch_idx, len(pbar))
                for k, v in loss_info.items():
                    if not(k in _loss_terms.keys()):
                        _loss_terms.update({k: v.item()})
                    else:
                        _loss_terms.update({k: v.item()+_loss_terms[k]})
                    print_info += k + ': {:.5f}; '.format(v.item())

                print_info += 'Total Loss: {:.5f}'.format(total_loss.item())
                pbar.set_description(print_info)

            _loss_average = 0.0
            if self.verborrea:
                print('-'*10 + 'val' + '-'*10)
                for k, v in _loss_terms.items():
                    print(k+'(Sum): ',  v)
                print('Loss (Sum): ', _loss)
                print(' ')
                for k, v in _loss_terms.items():
                    print(k+'(Avg): ',  v/len(DataSet))
                    _loss_average += v/len(DataSet)
                print('Loss (Avg): ', _loss_average)
                print(' ')

                with open(self.log_dir, 'a') as f:
                    f.write('-'*10 + 'train' + '-'*10 + '\n')
                    for k, v in _loss_terms.items():
                        f.write('{:s}  (Avg): {:.7f}'.format(k, v/len(DataSet))+'\n')
                    f.write('Loss (Avg): {:.7f}'.format(_loss_average)+'\n')
                    f.write('\n')

        return _loss_average, 0.0


    def update_list_lr(self,optimizer):
        list_lr=[]
        for param in optimizer.param_groups:
            list_lr.append(param['lr'])
        max_lr=max(list_lr)
        return max_lr, list_lr

