from numpy.lib.nanfunctions import nanpercentile
import torch
from torch import nn
from train import get_model, load_weights
import numpy as np
from functools import partial
from src.augmentations import none_aug
from flexpand import Expander
import os
import tifffile
from medpy.io import save as medsave
from medpy.io import load as medload
import nibabel as nib
from itertools import islice
import ast

import hydra
from omegaconf import DictConfig
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from tqdm.auto import tqdm

from metrics import get_labels, convert_tasks, postpocess, dice_score, compute_HD95, compute_clf_metrics

val_Dice, val_HD, val_acc, val_rec, val_prec, count = None, None, None, None, None, None
logits_dict, labels_dict, threshold_dict = defaultdict(list), defaultdict(list), dict()
n_sample = 50

# what to configure:
# network loading config (this will share loading function with train)
# configuration for the flexpand (this could be loaded as a separate configuration)
# processing parameters (batch size, activation function)
# configuration for the postfix of file to save (what should we do if none provided, maybe required?)

internal_medload = lambda addr: medload(addr)[0]


class Segmenter:
    def __init__(self, model_config, checkpoint_config, processing_parameters):
        self.model = get_model(**model_config)
        load_weights(self.model, **checkpoint_config)
        self.model = nn.DataParallel(self.model)
        self.model.to(torch.device('cuda:0'))
        self.model.eval()

        self.batch_size = processing_parameters['batch_size']

        if ('activation_function' in processing_parameters) and (
                processing_parameters['activation_function'] is not None):
            activation_function_name = processing_parameters['activation_function']
        else:
            activation_function_name = 'argmax'
        self.activation_function = partial(getattr(torch, activation_function_name), dim=1)

        if ('image_grain' in processing_parameters) and (processing_parameters['image_grain'] is not None):
            self.image_grain = processing_parameters['image_grain']
        else:
            self.image_grain = 32

        if 'dtype' in processing_parameters:
            self.output_dtype = processing_parameters['dtype']
        else:
            self.output_dtype = None

    def process_one_volume(self, volume, tasks):
        predictions = [[] for _ in tasks]
        for b in range(int(np.ceil(len(volume) / self.batch_size))):
            batch = volume[self.batch_size * b: self.batch_size * (b + 1)]
            #  batch = batch[:, :batch.shape[1] // self.image_grain * self.image_grain,
            #        :batch.shape[2] // self.image_grain * self.image_grain]
            batch = none_aug(image=batch)['image']
            batch = torch.from_numpy(batch[:, None, ...])
            batch = batch.to(torch.device('cuda:0'))
            embedding = self.model.module.get_embeddings(batch)
            for i, task in enumerate(tasks):
                pred = self.model.module.predict_head(embedding, task)
                # pred = self.activation_function(pred)
                pred = pred.detach().cpu().numpy()
                predictions[i].append(pred)

        for i in range(len(tasks)):
            predictions[i] = np.concatenate(predictions[i])

            '''
            if full_pred.ndim == 4:
                predictions[i] = np.pad(full_pred, ((0, 0), (0, 0), (0, volume.shape[1] - full_pred.shape[2]),
                                                   (0, volume.shape[2] - full_pred.shape[3])))
            elif full_pred.ndim == 3:
                predictions[i] = np.pad(full_pred, (
                    (0, 0), (0, volume.shape[1] - full_pred.shape[1]), (0, volume.shape[2] - full_pred.shape[2])))

            if self.output_dtype is not None:
                predictions[i] = predictions[i].astype(self.output_dtype)
            '''
        return predictions


def generate_in_out_pairs(data_config, saving_config, labels_path, tasks_path):
    exp = Expander()
    in_list = exp(**data_config)
    with open(tasks_path, 'r') as f:
        task_list = list(f.readlines())

    with open(labels_path, 'r') as f:
        label_list = list(f.readlines())

    tasks_list, labels_list, out_list = [], [], []
    for old_name, label, task in zip(in_list, label_list, task_list):
        if ('name' in saving_config) and (saving_config['name'] is not None):
            head, tail = os.path.split(old_name)
            tail = saving_config['name'] + str(task) + '.' + tail.split('.')[-1]
            out_list.append(os.path.join(head, tail))
        elif ('prefix' in saving_config) and (saving_config['prefix'] is not None):
            head, tail = os.path.split(old_name)
            tail = saving_config['prefix'] + '_' + tail
            out_list.append(os.path.join(head, tail))
        else:
            raise ValueError('Either name or prefix should be configured for saving. Overwrite was never an option!')
        tasks_list.append(convert_tasks(task))
        labels_list.append(label.strip())
    return list(zip(in_list, out_list, labels_list, tasks_list))


def load_process_save(segmenter, input_addr, label_addr, output_addr, tasks, mode='mots'):
    global val_Dice, val_HD, val_acc, val_rec, val_prec, count
    imgNII = nib.load(input_addr)
    labelNII = nib.load(label_addr)
    img = imgNII.get_fdata()
    label = labelNII.get_fdata()

    preds = segmenter.process_one_volume(img, tasks)
    preds = postpocess(preds, tasks)
    labels = get_labels(label, tasks)
    if mode == 'mots':
        for (pred, label, task) in zip(preds, labels, tasks):
            dice = dice_score(pred, label)
            HD = compute_HD95(label, pred)
            acc, rec, prec = compute_clf_metrics(label, pred)
            val_Dice[task] += dice
            val_HD[task] += HD
            val_acc[task] += acc
            val_rec[task] += rec
            val_prec[task] += prec
            count[task] += 1
    '''
    for i, task in enumerate(tasks):
        head, tail = os.path.split(output_addr)
        tail = f'{task}_{tail}'
        task_addr = os.path.join(head, tail)
        if input_addr.endswith('.tif') or input_addr.endswith('.tiff'):
            tifffile.imwrite(task_addr, preds[i])
        else:
            medsave(preds[i], task_addr)
    '''


@hydra.main(config_path='inference_configs', config_name="config")
def inference(cfg: DictConfig) -> None:

    global val_Dice, val_HD, val_acc, val_rec, val_prec, count
    n_tasks = cfg['n_tasks']
    val_Dice = np.zeros(n_tasks * 2)
    val_HD = np.zeros(n_tasks * 2)
    val_acc, val_prec, val_rec = np.zeros(n_tasks * 2), np.zeros(n_tasks * 2), np.zeros(n_tasks * 2)

    count = np.zeros(n_tasks * 2)

    seger = Segmenter(cfg['model'], cfg['checkpoint'], cfg['processing'])
    pairs = generate_in_out_pairs(cfg['dataset']['source'], cfg['dataset']['destination'], cfg['dataset']['labels'], cfg['dataset']['tasks'])
    for inp_addr, outp_addr, label_addr, tasks in tqdm(islice(pairs, 0, 5)):
        load_process_save(seger, inp_addr, label_addr, outp_addr, sorted(tasks))

    count[count == 0] = 1
    val_Dice /= count
    val_HD /= count
    val_acc /= count
    val_rec /= count
    val_prec /= count

    print("Sum results")
    for t in range(n_tasks):
        print(f'Sum: Task: {t // 2}\n'
              f'Organ_Dice:{val_Dice[t * 2]:.4f} Tumor_Dice:{val_Dice[t * 2 + 1]:.4f}\n'
              f'Organ_HD:{val_HD[t * 2]:.4f} Tumor_HD:{val_HD[t * 2 + 1]:.4f}\n'
              f'Organ acc, rec, prec:{val_acc[t * 2]:.2f}, {val_rec[t * 2]:.2f}, {val_prec[t * 2]:.4f} '
              f'Tumor acc, rec, prec:{val_acc[t * 2 + 1]:.2f}, {val_rec[t * 2 + 1]:.2f}, {val_prec[t * 2 + 1]:.4f}\n')


from numpy.lib.nanfunctions import nanpercentile
import torch
from torch import nn
from train import get_model, load_weights
import numpy as np
from functools import partial
from src.augmentations import none_aug
from flexpand import Expander
import os
import tifffile
from medpy.io import save as medsave
from medpy.io import load as medload
import nibabel as nib
from itertools import islice

import hydra
from omegaconf import DictConfig
from tqdm.auto import tqdm

from metrics import get_labels, convert_tasks, postpocess, dice_score, compute_HD95, compute_clf_metrics

val_Dice, val_HD, val_acc, val_rec, val_prec, count = None, None, None, None, None, None


# what to configure:
# network loading config (this will share loading function with train)
# configuration for the flexpand (this could be loaded as a separate configuration)
# processing parameters (batch size, activation function)
# configuration for the postfix of file to save (what should we do if none provided, maybe required?)

internal_medload = lambda addr: medload(addr)[0]


class Segmenter:
    def __init__(self, model_config, checkpoint_config, processing_parameters):
        self.model = get_model(**model_config)
        load_weights(self.model, **checkpoint_config)
        self.model = nn.DataParallel(self.model)
        self.model.to(torch.device('cuda:0'))
        self.model.eval()

        self.batch_size = processing_parameters['batch_size']

        if ('activation_function' in processing_parameters) and (
                processing_parameters['activation_function'] is not None):
            activation_function_name = processing_parameters['activation_function']
        else:
            activation_function_name = 'argmax'
        self.activation_function = partial(getattr(torch, activation_function_name), dim=1)

        if ('image_grain' in processing_parameters) and (processing_parameters['image_grain'] is not None):
            self.image_grain = processing_parameters['image_grain']
        else:
            self.image_grain = 32

        if 'dtype' in processing_parameters:
            self.output_dtype = processing_parameters['dtype']
        else:
            self.output_dtype = None

    def process_one_volume(self, volume, tasks, logits=True):
        predictions = [[] for _ in tasks]
        for b in range(int(np.ceil(len(volume) / self.batch_size))):
            batch = volume[self.batch_size * b: self.batch_size * (b + 1)]
            batch = none_aug(image=batch)['image']
            batch = torch.from_numpy(batch[:, None, ...])
            batch = batch.to(torch.device('cuda:0'))
            embedding = self.model.module.get_embeddings(batch)
            for i, task in enumerate(tasks):
                pred = self.model.module.predict_head(embedding, task, logits=logits)
                pred = pred.detach().cpu().numpy()
                predictions[i].append(pred)

        for i in range(len(tasks)):
            predictions[i] = np.concatenate(predictions[i])
        return predictions


def generate_in_out_pairs(data_config, saving_config, labels_path, tasks_path):
    exp = Expander()
    in_list = exp(**data_config)
    with open(tasks_path, 'r') as f:
        task_list = list(f.readlines())

    with open(labels_path, 'r') as f:
        label_list = list(f.readlines())

    tasks_list, labels_list, out_list = [], [], []
    for old_name, label, task in zip(in_list, label_list, task_list):
        if ('name' in saving_config) and (saving_config['name'] is not None):
            head, tail = os.path.split(old_name)
            tail = saving_config['name'] + str(task) + '.' + tail.split('.')[-1]
            out_list.append(os.path.join(head, tail))
        elif ('prefix' in saving_config) and (saving_config['prefix'] is not None):
            head, tail = os.path.split(old_name)
            tail = saving_config['prefix'] + '_' + tail
            out_list.append(os.path.join(head, tail))
        else:
            raise ValueError('Either name or prefix should be configured for saving. Overwrite was never an option!')
        tasks_list.append(ast.literal_eval(task))
        labels_list.append(label.strip())
    return list(zip(in_list, out_list, labels_list, tasks_list))


def load_process_save(segmenter, input_addr, label_addr, output_addr, tasks, mode='mots'):
    global val_Dice, val_HD, val_acc, val_rec, val_prec, count
    imgNII = nib.load(input_addr)
    labelNII = nib.load(label_addr)
    img = imgNII.get_fdata()
    label = labelNII.get_fdata()

    preds = segmenter.process_one_volume(img, tasks)
    preds = postpocess(preds, tasks)
    labels = get_labels(label, tasks)
    if mode == 'mots':
        for (pred, label, task) in zip(preds, labels, tasks):
            dice = dice_score(pred, label)
            HD = compute_HD95(label, pred)
            acc, rec, prec = compute_clf_metrics(label, pred)
            val_Dice[task] += dice
            val_HD[task] += HD
            val_acc[task] += acc
            val_rec[task] += rec
            val_prec[task] += prec
            count[task] += 1
    '''
    for i, task in enumerate(tasks):
        head, tail = os.path.split(output_addr)
        tail = f'{task}_{tail}'
        task_addr = os.path.join(head, tail)
        if input_addr.endswith('.tif') or input_addr.endswith('.tiff'):
            tifffile.imwrite(task_addr, preds[i])
        else:
            medsave(preds[i], task_addr)
    '''


def collect_preds_n_labels(segmenter, input_addr, label_addr, tasks):
    global logits, labels
    imgNII = nib.load(input_addr)
    labelNII = nib.load(label_addr)
    img = imgNII.get_fdata()
    label = labelNII.get_fdata()

    preds = segmenter.process_one_volume(img, tasks, logits=True)
    labels = get_labels(label, tasks)
    for (pred, label, task) in zip(preds, labels, tasks):
        pred, label = pred.reshape(-1), label.reshape(-1)
        idx = np.random.choice(len(pred), n_sample)
        logits_dict[task].append(pred[idx])
        labels_dict[task].append(labels[idx])


@hydra.main(config_path='inference_configs', config_name="config")
def inference(cfg: DictConfig) -> None:

    global val_Dice, val_HD, val_acc, val_rec, val_prec, count
    n_tasks = cfg['n_tasks']
    val_Dice = np.zeros(n_tasks * 2)
    val_HD = np.zeros(n_tasks * 2)
    val_acc, val_prec, val_rec = np.zeros(n_tasks * 2), np.zeros(n_tasks * 2), np.zeros(n_tasks * 2)

    count = np.zeros(n_tasks * 2)

    seger = Segmenter(cfg['model'], cfg['checkpoint'], cfg['processing'])
    pairs = generate_in_out_pairs(cfg['dataset']['source'], cfg['dataset']['destination'], cfg['dataset']['labels'], cfg['dataset']['tasks'])
    for inp_addr, outp_addr, label_addr, tasks in tqdm(islice(pairs, 0, 5)):
        load_process_save(seger, inp_addr, label_addr, outp_addr, sorted(tasks))

    count[count == 0] = 1
    val_Dice /= count
    val_HD /= count
    val_acc /= count
    val_rec /= count
    val_prec /= count

    print("Sum results")
    for t in range(n_tasks):
        print(f'Sum: Task: {t // 2}\n'
              f'Organ_Dice:{val_Dice[t * 2]:.4f} Tumor_Dice:{val_Dice[t * 2 + 1]:.4f}\n'
              f'Organ_HD:{val_HD[t * 2]:.4f} Tumor_HD:{val_HD[t * 2 + 1]:.4f}\n'
              f'Organ acc, rec, prec:{val_acc[t * 2]:.2f}, {val_rec[t * 2]:.2f}, {val_prec[t * 2]:.4f} '
              f'Tumor acc, rec, prec:{val_acc[t * 2 + 1]:.2f}, {val_rec[t * 2 + 1]:.2f}, {val_prec[t * 2 + 1]:.4f}\n')


@hydra.main(config_path='threshold_configs', config_name="config")
def get_thresholds(cfg: DictConfig) -> None:
    seger = Segmenter(cfg['model'], cfg['checkpoint'], cfg['processing'])
    pairs = generate_in_out_pairs(cfg['dataset']['source'], cfg['dataset']['destination'], cfg['dataset']['labels'],
                                  cfg['dataset']['tasks'])
    for inp_addr, outp_addr, label_addr, tasks in tqdm(islice(pairs, 0, 5)):
        collect_preds_n_labels(seger, inp_addr, label_addr, sorted(tasks))
    for task in logits_dict.keys():
        logits_dict[task] = np.concatenate(logits_dict[task])
        labels_dict[task] = np.concatenate(labels_dict[task])
    for task in logits_dict.keys():
        precision, recall, thresholds = precision_recall_curve(labels_dict[task],
                                                               logits_dict[task])
        f1 = 2 * precision * recall / (precision + recall)
        i = np.argmax(f1)
        threshold_dict[task] = thresholds[i]
    print('Thresholds:')
    for task in logits_dict.keys():
        print(f'Task: {task}, Threshold: {threshold_dict[task]}')


if __name__ == "__main__":
    #inference()
    get_thresholds()