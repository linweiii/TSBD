

import argparse
import copy
import os,sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from tqdm import tqdm

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from defense.base import defense
from torch.utils.data import DataLoader, RandomSampler, random_split
from collections import OrderedDict

from utils.aggregate_block.train_settings_generate import argparser_criterion
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from utils.choose_index import choose_index
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
import csv

method_name = 'tsbd'
reinit_ratio = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]


def zero_reinit_weight(net, top_num, changed_values_neuron, n2w_dict, wratio):
    state_dict = net.state_dict()
    merge_list = []
    for layer_name, nidx, value in changed_values_neuron[:top_num]:
        mn = mixed_name(layer_name, nidx)
        merge_list += n2w_dict[mn]
    reinit_list = sorted(merge_list, reverse=True)[:int(len(merge_list)*wratio)]
    min_reinit_weight = min(reinit_list)
    for layer_name, nidx, value in changed_values_neuron[:top_num]:
        mn = mixed_name(layer_name, nidx)
        reinit_weight_index = [int(index) for index, weight_value in enumerate(n2w_dict[mn]) if weight_value >= min_reinit_weight]
        state_dict[layer_name][int(nidx)].view(-1)[reinit_weight_index] = 0.0
        
    net.load_state_dict(state_dict)
    return len(reinit_list)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    values = list(zip(layer, idx, value))
    return values

def get_layerName_from_type(model, layer_type):
    if layer_type == 'conv':
        instance_name = nn.Conv2d
    elif layer_type == 'bn':
        instance_name = nn.BatchNorm2d
    else:
        raise SystemError('NO valid layer_type match!')
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, instance_name) and 'shortcut' not in name:
            layer_names.append(name+'.weight')
    return layer_names


def mixed_name(layer_name, idx):
    return layer_name+'.'+str(idx)


class tsbd(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        self.policyLoss = nn.CrossEntropyLoss(reduction='none')

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--clean_file', type=str, help='the location of clean model')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr_un', type=float)
        parser.add_argument('--lr_ft', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/tsbd/config.yaml", help='the path of yaml')
        parser.add_argument('--layer_type', type=str, help='the type of layer for reinitialization')
        # parser.add_argument('--reinit_num', type=int)
        parser.add_argument('--m', type=float)
        parser.add_argument('--ft_epoch', type=int)
        parser.add_argument('--r', type=float, help='the r for regularization')
        parser.add_argument('--alpha', type=float, help='the alpha for regularization')

        parser.add_argument('--model_type', choices=['bd', 'clean'], help='whether the model is BD model')
        parser.add_argument('--data_type', choices=['clean_test','poison_test','clean_val'], help='the unlearning data type')
        parser.add_argument('--record_layer', type=str, help='the layer name for record')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/{method_name}/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + f'checkpoint_{self.args.model_type}_{self.args.data_type}/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        self.device = self.args.device


    def train_unlearning(self, args, model, criterion, optimizer, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        gradNorm = []
        pbar = tqdm(data_loader)
        for i, (images, labels, *additional_info)in enumerate(pbar):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()

            (-loss).backward()

            record_layer_param = [param for name,param in model.named_parameters() if name == args.record_layer][0]
            record_layer_param_grad = record_layer_param.grad.view(record_layer_param.shape[0],-1).abs().sum(dim=-1)
            gradNorm.append(record_layer_param_grad)

            optimizer.step()
            pbar.set_description("Loss: "+str(loss))

        gradNorm = torch.stack(gradNorm,0).float()
        avg_gradNorm = gradNorm.mean(dim=0)
        var_gradNorm = gradNorm.var(dim=0)
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc, avg_gradNorm, var_gradNorm


    def train_finetuning_reg(self, args, model, criterion, optimizer, data_loader, r, alpha):
        model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (images, labels, *additional_info) in enumerate(data_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            nb_samples += images.size(0)

            model_temp = copy.deepcopy(model)
            out1 = model_temp(images)
            loss1 = criterion(out1, labels)
            model_temp.zero_grad()
            loss1.backward()
            g1 = [param.grad.data.clone() for param in model_temp.parameters()]

            with torch.no_grad():
                for param, grad in zip(model_temp.parameters(), g1):
                    param.data += r * (grad/grad.norm())
            out2 = model_temp(images)
            loss2 = criterion(out2, labels)
            model_temp.zero_grad()
            loss2.backward()
            g2 = [param.grad.data.clone() for param in model_temp.parameters()]
            
            optimizer.zero_grad()
            final_gradients = [(1 - alpha) * g1_item + alpha * g2_item for g1_item, g2_item in zip(g1, g2)]
            for param, grad in zip(model.parameters(), final_gradients):
                if grad is not None:
                    param.grad = grad
            optimizer.step()

            output = model(images)
            loss = criterion(output, labels)
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc
    
    def test(self, args, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels, *add_info) in enumerate(data_loader):
                images, labels = images.to(args.device), labels.to(args.device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc
        
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        #load clean val set
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length) 
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_clean = self.result['clean_train']
        data_set_clean.wrapped_dataset = data_set_without_tran
        data_set_clean.wrap_img_transform = train_tran
        # data_set_clean.wrapped_dataset.getitem_all = False

        clean_val_loader = DataLoader(data_set_clean, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
        test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

        criterion = argparser_criterion(args)
        is_BDmodel = [True if self.args.model_type == 'bd' else False][0]
        data_type = self.args.data_type
        print("*** is_BDmodel:", is_BDmodel)
        print("*** data_type:", data_type)

        try:
            model_clean = generate_cls_model(self.args.model,self.args.num_classes)
            model_clean.load_state_dict(torch.load(f'./record/{self.args.clean_file}/clean_model.pth'))
            model_clean = model_clean.to(args.device)
        except:
            print("No clean model loaded.")
        
        model_ori = generate_cls_model(self.args.model,self.args.num_classes)
        model_ori.load_state_dict(self.result['model'])
        model_ori = model_ori.to(args.device)

        if is_BDmodel:
            model = copy.deepcopy(model_ori)
            parameters_o = list(model_ori.named_parameters())
        else:
            model = copy.deepcopy(model_clean)
            parameters_o = list(model_clean.named_parameters())
        
        target_layers = get_layerName_from_type(model_ori, args.layer_type)
        params_o = {'names':[n for n, v in parameters_o if n in target_layers],
                    'params':[v for n, v in parameters_o if n in target_layers]}
        
        _, val_acc = self.test(args, model, criterion, clean_val_loader)
        _, test_acc = self.test(args, model, criterion, data_clean_loader)
        _, test_asr = self.test(args, model, criterion, data_bd_loader)
        logging.info(f"Test loaded model: acc_{test_acc}, asr_{test_asr}, val_acc_{val_acc}")

        do_unlearn = True
        if do_unlearn:
            unlearn_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_un, momentum=0.9)

            csv_grad_avg = os.path.join(args.checkpoint_save, f'grad_avg_{args.record_layer}.csv')
            csv_grad_var = os.path.join(args.checkpoint_save, f'grad_var_{args.record_layer}.csv')
            with open(csv_grad_avg, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['Epoch', 'train_loss', 'train_acc', 'test_acc', 'test_asr', 'val_acc'] + [f'neuron_{i}' for i in range([param for name,param in model.named_parameters() if name == args.record_layer][0].shape[0])]
                writer.writerow(header)
            with open(csv_grad_var, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['Epoch', 'train_loss', 'train_acc', 'test_acc', 'test_asr', 'val_acc'] + [f'neuron_{i}' for i in range([param for name,param in model.named_parameters() if name == args.record_layer][0].shape[0])]
                writer.writerow(header)


            logging.info("Unlearning...")
            for epoch in range(args.epochs):
                if data_type == 'clean_val':
                    train_loss, train_acc, avg_gradNorm, var_gradNorm = self.train_unlearning(args, model, criterion, unlearn_optimizer, clean_val_loader)
                elif data_type == 'clean_test':
                    train_loss, train_acc, avg_gradNorm, var_gradNorm = self.train_unlearning(args, model, criterion, unlearn_optimizer, data_clean_loader)
                elif data_type == 'poison_test':
                    train_loss, train_acc, avg_gradNorm, var_gradNorm = self.train_unlearning(args, model, criterion, unlearn_optimizer, data_bd_loader)
                logging.info(f"{epoch} Train unlearned model: train_loss_{train_loss}, train_acc_{train_acc}")

                _, val_acc = self.test(args, model, criterion, clean_val_loader)
                _, test_acc = self.test(args, model, criterion, data_clean_loader)
                _, test_asr = self.test(args, model, criterion, data_bd_loader)
                logging.info(f"{epoch} Test unlearned model: acc_{test_acc}, asr_{test_asr}, val_acc_{val_acc}")
                logging.info('-'*50)
                
                with open(csv_grad_avg, 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = [epoch, train_loss, train_acc, test_acc, test_asr, val_acc] + avg_gradNorm.tolist()
                    writer.writerow(row)
                with open(csv_grad_var, 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = [epoch, train_loss, train_acc, test_acc, test_asr, val_acc] + var_gradNorm.tolist()
                    writer.writerow(row)

                if data_type == 'clean_val' and val_acc <= 0.10:
                    logging.info(f"Break unlearn.")
                    break
                elif data_type == 'clean_test' and test_acc <= 0.10:
                    logging.info(f"Break unlearn.")
                    break
                elif data_type == 'poison_test' and test_asr <= 0.05:
                    logging.info(f"Break unlearn.")
                    break

                
            parameters_u = list(model.named_parameters())
            params_u = {'names':[n for n, v in parameters_u if n in target_layers],
                            'params':[v for n, v in parameters_u if n in target_layers]}

            changed_values_neuron = []
            changed_values_weightOrder = {}
            count = 0
            for layer_i in range(len(params_u['params'])):
                name_i  = params_u['names'][layer_i]
                changed_params_i = params_u['params'][layer_i] - params_o['params'][layer_i]
                changed_weight_i = changed_params_i.view(changed_params_i.shape[0], -1).abs()
                changed_neuron_i = changed_weight_i.sum(dim=-1)
                for idx in range(changed_neuron_i.size(0)):
                    neuron_name =  mixed_name(name_i, idx)
                    changed_values_weightOrder[neuron_name] = changed_weight_i[idx].tolist()
                    changed_values_neuron.append('{} \t {} \t {} \t {:.4f} \n'.format(count, name_i, idx, changed_neuron_i[idx].item()))
                    count += 1
            with open(os.path.join(args.checkpoint_save, f'nwc.txt'), "w") as f:
                f.write('No \t Layer_Name \t Neuron_Idx \t Score \n')
                f.writelines(changed_values_neuron)
            torch.save(changed_values_weightOrder, os.path.join(args.checkpoint_save, 'n2w_dict.pt'))
            torch.save(model.state_dict(), os.path.join(args.checkpoint_save, 'unlearned_model.pt'))

                
        # ==================
        max2min = True
        changed_values_neuron = read_data(args.checkpoint_save + f'nwc.txt')
        changed_values_neuron = sorted(changed_values_neuron, key=lambda x: float(x[2]), reverse=max2min)

        n2w_dict = torch.load(os.path.join(args.checkpoint_save, 'n2w_dict.pt'))

        agg = Metric_Aggregator()
        ft_agg = Metric_Aggregator()
        logging.info("Reinitializing...")
        
        total_num = len(changed_values_neuron)

        for ratio in reinit_ratio:
            top_num = int(total_num*ratio)
            if is_BDmodel:
                model_copy = copy.deepcopy(model_ori)
            else:
                model_copy = copy.deepcopy(model_clean)

            reinit_weight_num = zero_reinit_weight(model_copy, top_num, changed_values_neuron, n2w_dict, args.m)

            self.set_trainer(model_copy)
            self.trainer.set_with_dataloader(
                ### the train_dataload has nothing to do with the backdoor defense
                train_dataloader = test_dataloader_dict['bd_test_dataloader'],
                test_dataloader_dict = test_dataloader_dict,

                criterion = criterion,
                optimizer = None,
                scheduler = None,
                device = self.args.device,
                amp = self.args.amp,

                frequency_save = self.args.frequency_save,
                save_folder_path = self.args.save_path,
                save_prefix = 'reinitialize',

                prefetch = self.args.prefetch,
                prefetch_transform_attr_name = "ori_image_transform_in_loading",
                non_blocking = self.args.non_blocking,


                )
            clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.trainer.test_current_model(
                test_dataloader_dict, args.device,
            )
            logging.info(f"Test reinitialized model: acc_{test_acc}, asr_{test_asr}, ra_{test_ra}")

            agg({
                    "n": ratio,
                    "n_number": top_num,
                    "m": args.m,
                    "reinit_weight_num": reinit_weight_num,
                    "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                    "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                    "test_acc": test_acc,
                    "test_asr": test_asr,
                    "test_ra": test_ra,
                })
            agg.to_dataframe().to_csv(os.path.join(args.checkpoint_save, f'stage1_df_m{args.m}.csv'))

            is_finetune = True
            if is_finetune:
                logging.info("Fine Tuning...")

                update_neuron_params_optimizer = torch.optim.SGD(model_copy.parameters(), lr=args.lr_ft, momentum=0.9)

                pbar = tqdm(range(args.ft_epoch+1))
                for epoch in pbar:
                    self.train_finetuning_reg(args, model_copy, criterion, update_neuron_params_optimizer, clean_val_loader, self.args.r, self.args.alpha)

                    if epoch % 10 == 0:
                        self.set_trainer(model_copy)
                        self.trainer.set_with_dataloader(
                            ### the train_dataload has nothing to do with the backdoor defense
                            train_dataloader = test_dataloader_dict['bd_test_dataloader'],
                            test_dataloader_dict = test_dataloader_dict,

                            criterion = criterion,
                            optimizer = None,
                            scheduler = None,
                            device = self.args.device,
                            amp = self.args.amp,

                            frequency_save = self.args.frequency_save,
                            save_folder_path = self.args.save_path,
                            save_prefix = 'finetune',

                            prefetch = self.args.prefetch,
                            prefetch_transform_attr_name = "ori_image_transform_in_loading",
                            non_blocking = self.args.non_blocking,
                            )
                        clean_test_loss_avg_over_batch, \
                                bd_test_loss_avg_over_batch, \
                                test_acc, \
                                test_asr, \
                                test_ra = self.trainer.test_current_model(
                            test_dataloader_dict, args.device,
                        )
                        logging.info(f"Test finetuned model: acc_{test_acc}, asr_{test_asr}, ra_{test_ra}")
                
                        ft_agg({
                                "finetune_epoch":epoch,
                                "n": ratio,
                                "n_number": top_num,
                                "m": args.m,
                                "r": args.r,
                                "alpha": args.alpha,
                                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                                "test_acc": test_acc,
                                "test_asr": test_asr,
                                "test_ra": test_ra,
                            })
                        ft_agg.to_dataframe().to_csv(os.path.join(args.checkpoint_save, f'stage2_df_ft{args.ft_epoch}_r{args.r}_alpha{args.alpha}_m{args.m}.csv'))

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    tsbd.add_arguments(parser)
    args = parser.parse_args()
    method = tsbd(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = method.defense(args.result_file)