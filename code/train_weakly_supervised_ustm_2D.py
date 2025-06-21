import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import torch.nn.functional as F

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds
from skimage import feature as fe
from utils.supcon_loss import SupConLoss
import cv2
import math
import random
import torch.nn.functional as F
from scipy.ndimage.interpolation import zoom
from skimage import morphology, measure
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt as dist_tranform
import glob
import json
import cv2 as cv
from scipy import ndimage as ndi_morph
from skimage import measure

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/pCE_Seg_USTM', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--lambda_value', type=float,  default=0.6, help='lambda value')
parser.add_argument('--use_aa', action='store_true',
                    default=False, help='use_aa')
parser.add_argument('--use_cluster', action='store_true',
                    default=False, help='use_cluster')
parser.add_argument('--use_contrast', action='store_true',
                    default=False, help='use_contrast')
parser.add_argument('--contrast_loss_weight', type=float,  default=0.3, help='contrast_loss_weight')
parser.add_argument('--div_nums', type=int,  default=8, help='div_nums')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * ramps.sigmoid_rampup(epoch, 60)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def one_hot(inputs, n_classes, use_gpu=True):
    inputs = inputs.long().unsqueeze(1)

    target_shape = list(inputs.shape)
    target_shape[1] = n_classes

    output = torch.zeros(target_shape)
    if use_gpu:
        output = output.cuda()

    return output.scatter_(1, inputs, 1)


def propagation(x, class_feature):
    m_batchsize, C, height, width = x.size()
    m_batchsize1, C1, height1, width1 = class_feature.size()
    proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)
    proj_key = class_feature.view(m_batchsize1, -1, width1 * height1)
    energy = torch.bmm(proj_query, proj_key)
    attention = F.softmax(energy, dim=-1)
    proj_value = class_feature.view(m_batchsize1, -1, width1 * height1)

    out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    out = out.view(m_batchsize, C, height, width)

    return out

def cal_equal(input):
    n_tensor = len(input)
    base = torch.ones(input[0].shape).cuda()
    for i in range(n_tensor-1):
        is_equal = torch.eq(input[i], input[i+1])
        base = base * is_equal
    return base

def Sobel(img):

    x = cv2.Sobel(img*255, cv2.CV_64F, 1, 0)
    y = cv2.Sobel(img*255, cv2.CV_64F, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return (Sobel - Sobel.min()) / (Sobel.max() - Sobel.min())

def Prewitt(img):

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(img*255, cv2.CV_64F, kernelx)
    y = cv2.filter2D(img*255, cv2.CV_64F, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return (Prewitt - Prewitt.min()) / (Prewitt.max() - Prewitt.min())

def Roberts(img):

    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(img*255, cv2.CV_64F, kernelx)
    y = cv2.filter2D(img*255, cv2.CV_64F, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return (Roberts - Roberts.min()) / (Roberts.max() - Roberts.min())

def Laplacian(img):

    dst = cv2.Laplacian(img*255, cv2.CV_64F, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    return (Laplacian - Laplacian.min()) / (Laplacian.max() - Laplacian.min())

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                                class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    seg_ce_loss = CrossEntropyLoss(ignore_index=-1)
    contrast_loss = SupConLoss()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, feature, outputs_binary = model(volume_batch)


            #rot_times = random.randrange(0, 4)
            # rotated_volume_batch = torch.rot90(volume_batch, rot_times, [2, 3])
            noise = torch.clamp(torch.randn_like(
                volume_batch) * 0.1, -0.05, 0.05)
            with torch.no_grad():
                ema_inputs = volume_batch + noise
                ema_output, _, ema_output_binary = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
                pseudo_label = torch.argmax(ema_output_soft, dim=1)
                if args.use_cluster:
                    ema_output_soft_binary = torch.softmax(ema_output_binary, dim=1)
                    pseudo_label_binary = torch.argmax(ema_output_soft_binary, dim=1)


            if args.use_aa:
                import copy
                volume_batch1 = copy.deepcopy(volume_batch)
                label_batch1 = copy.deepcopy(label_batch)
                pseudo_label1 = copy.deepcopy(pseudo_label)
                if args.use_cluster:
                    import copy
                    import cv2
                    label_batch_binary1 = copy.deepcopy(label_batch).int()
                    label_batch_binary1[label_batch_binary1 == 4] = -1
                    label_batch_binary1[label_batch_binary1 > 0] = 1
                    label_batch_binary1[label_batch_binary1 == -1] = 4
                    pseudo_label_binary1 = copy.deepcopy(pseudo_label_binary)
                n, c, h, w = volume_batch1.shape
                data_volume = []
                data_label = []
                data_pseudo = []
                if args.use_cluster:
                    data_label_binary = []
                    data_pseudo_binary = []
                for i in range(n):
                    random_number_high = random.randint(h // 3, h // 2)
                    random_number_width = random.randint(w // 3, w // 2)
                    random_number_resize = random.randint(0, n)
                    mask_temp = (label_batch1[i] < 4).cpu().numpy()
                    voxels = np.where(mask_temp != 0)
                    # print(mask_temp.shape, voxels.shape)
                    if len(voxels[0]) == 0:
                        minXidx = 0
                        maxXidx = 0
                        minYidx = 0
                        maxYidx = 0
                        # print(mask_temp.shape, voxels.shape)
                    else:
                        minXidx = int(np.min(voxels[0]))
                        maxXidx = int(np.max(voxels[0]))
                        minYidx = int(np.min(voxels[1]))
                        maxYidx = int(np.max(voxels[1]))
                    volume_batch_crop = volume_batch1[i, 0, minXidx:maxXidx, minYidx:maxYidx]
                    label_batch_crop = label_batch1[i, minXidx:maxXidx, minYidx:maxYidx]
                    pseudo_label_crop = pseudo_label1[i, minXidx:maxXidx, minYidx:maxYidx]
                    if args.use_cluster:
                        label_batch_binary_crop = label_batch_binary1[i, minXidx:maxXidx, minYidx:maxYidx]
                        pseudo_label_binary_crop = pseudo_label_binary1[i, minXidx:maxXidx, minYidx:maxYidx]
                    data_volume.append(volume_batch_crop.cuda())
                    data_label.append(label_batch_crop.cuda())
                    data_pseudo.append(pseudo_label_crop.cuda())
                    if args.use_cluster:
                        data_label_binary.append(label_batch_binary_crop.cuda())
                        data_pseudo_binary.append(pseudo_label_binary_crop.cuda())
                if args.use_cluster:
                    combined = list(zip(data_volume, data_label, data_pseudo, data_label_binary, data_pseudo_binary))
                    random.shuffle(combined)
                    data_volume, data_label, data_pseudo, data_label_binary, data_pseudo_binary = zip(*combined)
                else:
                    combined = list(zip(data_volume, data_label, data_pseudo))
                    random.shuffle(combined)
                    data_volume, data_label, data_pseudo = zip(*combined)

                for i in range(n):
                    random_number_choice = random.randint(0, n-1)
                    volume_batch_crop = data_volume[random_number_choice]
                    label_batch_crop = data_label[random_number_choice]
                    pseudo_label_crop = data_pseudo[random_number_choice]
                    id_num = 0
                    while volume_batch_crop.shape[-1] == 0 and id_num < args.batch_size:
                        random_number_choice = random.randint(0, n - 1)
                        volume_batch_crop = data_volume[random_number_choice]
                        label_batch_crop = data_label[random_number_choice]
                        pseudo_label_crop = data_pseudo[random_number_choice]
                        id_num += 1

                    if args.use_cluster:
                        label_batch_binary_crop = data_label_binary[random_number_choice]
                        pseudo_label_binary_crop = data_pseudo_binary[random_number_choice]
                    mask_temp = (label_batch1[i] < 4).cpu().numpy()
                    voxels = np.where(mask_temp != 0)
                    if len(voxels[0]) == 0:
                        minXidx = 0
                        maxXidx = 0
                        minYidx = 0
                        maxYidx = 0
                        # print(mask_temp.shape, voxels.shape)
                    else:
                        minXidx = int(np.min(voxels[0]))
                        maxXidx = int(np.max(voxels[0]))
                        minYidx = int(np.min(voxels[1]))
                        maxYidx = int(np.max(voxels[1]))
                    size_h = maxXidx - minXidx
                    size_w = maxYidx - minYidx
                    h_crop, w_crop = volume_batch_crop.shape
                    if size_h != 0 and size_w != 0 and id_num < args.batch_size:
                        volume_batch_crop = torch.from_numpy(zoom(volume_batch_crop.cpu().numpy(), (
                                size_h / h_crop, size_w / w_crop), order=0)).cuda()
                        label_batch_crop = torch.from_numpy(zoom(label_batch_crop.cpu().numpy(), (
                                size_h / h_crop, size_w / w_crop), order=0)).cuda()
                        pseudo_label_crop = torch.from_numpy(zoom(pseudo_label_crop.cpu().numpy(), (
                                size_h / h_crop, size_w / w_crop), order=0)).cuda()
                        volume_batch1[i, 0, minXidx: maxXidx, minYidx: maxYidx] = volume_batch_crop
                        label_batch1[i, minXidx: maxXidx, minYidx: maxYidx] = label_batch_crop
                        pseudo_label1[i, minXidx: maxXidx, minYidx: maxYidx] = pseudo_label_crop
                        if args.use_cluster:
                            label_batch_binary_crop = torch.from_numpy(zoom(label_batch_binary_crop.cpu().numpy(), (
                                size_h / h_crop, size_w / w_crop), order=0)).cuda()
                            pseudo_label_binary_crop = torch.from_numpy(zoom(pseudo_label_binary_crop.cpu().numpy(), (
                                size_h / h_crop, size_w / w_crop), order=0)).cuda()
                            label_batch_binary1[i, minXidx: maxXidx, minYidx: maxYidx] = label_batch_binary_crop
                            pseudo_label_binary1[i, minXidx: maxXidx, minYidx: maxYidx] = pseudo_label_binary_crop
                outputs1, feature1, outputs_binary1 = model(volume_batch1)
                loss_ce1 = ce_loss(outputs1, label_batch1.long())
                supervised_loss1 = loss_ce1
                seg_loss1 = seg_ce_loss(outputs1, pseudo_label1.long())
                loss1 = supervised_loss1 + args.lambda_value * seg_loss1
                if args.use_cluster:

                    loss_binary1 = ce_loss(outputs_binary1, label_batch_binary1.long()) + args.lambda_value * seg_ce_loss(
                        outputs_binary1, pseudo_label_binary1.long())
                    loss1+=loss_binary1


            if args.use_contrast:
                shape = volume_batch.shape
                img_size = shape[-1]
                div_num = args.div_nums
                block_size = math.ceil(img_size / div_num)
                mean_feature = torch.zeros(shape[0], div_num ** 2, feature.shape[1])
                feature_label = torch.zeros(shape[0], div_num ** 2)
                import copy
                label_scribble = copy.deepcopy(label_batch)
                for j in range(div_num):
                    for k in range(div_num):
                        block_feature = feature[:, :, j * block_size:(j + 1) * block_size,
                                        k * block_size:(k + 1) * block_size]
                        block_label = label_scribble[:, j * block_size:(j + 1) * block_size,
                                      k * block_size:(k + 1) * block_size]
                        mean_block_feture = torch.mean(block_feature, dim=[2, 3])
                        mean_block_feture = F.normalize(mean_block_feture, p=2, dim=1)
                        mean_feature[:, j * div_num + k] = mean_block_feture
                        block_label[block_label == 4] = 0
                        feature_label[:, j * div_num + k] = torch.sum(block_label, dim=[1, 2])

                feature_label[feature_label > 0] = 1

                mean_feature = mean_feature.unsqueeze(-1)
                feature_label = feature_label.unsqueeze(-1)
                new_features = mean_feature.unsqueeze(1).permute(0, 1, 3, 2, 4)
                new_labels = feature_label.unsqueeze(1)
                con_loss = contrast_loss(new_features, new_labels)
            loss_ce = ce_loss(outputs, label_batch.long())
            supervised_loss = loss_ce

            if args.use_contrast:
                if args.use_cluster:
                    if iter_num < 5000:
                        seg_loss = seg_ce_loss(outputs, pseudo_label_binary1.long()) + args.contrast_loss_weight * con_loss
                    else:
                        seg_loss = seg_ce_loss(outputs, pseudo_label.long()) + args.contrast_loss_weight * con_loss
                else:
                    seg_loss = seg_ce_loss(outputs, pseudo_label.long()) + args.contrast_loss_weight * con_loss
            else:
                if args.use_cluster:
                    if iter_num < 5000:
                        seg_loss = seg_ce_loss(outputs, pseudo_label_binary1.long())
                    else:
                        seg_loss = seg_ce_loss(outputs, pseudo_label.long()) 
                else:
                    seg_loss = seg_ce_loss(outputs, pseudo_label.long())

            loss = supervised_loss + args.lambda_value * seg_loss
            if args.use_cluster:
                loss+=loss_binary
            if args.use_aa:
                loss+=loss1



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f' %
                (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code_prostate'):
        shutil.rmtree(snapshot_path + '/code_prostate')
    shutil.copytree('.', snapshot_path + '/code_prostate',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
