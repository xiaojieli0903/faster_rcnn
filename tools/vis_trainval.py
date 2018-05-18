# --------------------------------------------------------
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient, vis_rois

from model.faster_rcnn_2s.vgg16 import vgg16
from model.faster_rcnn_2s.resnet_twostage import resnet
from model.nms.nms_wrapper import nms
from model.utils.net_utils import save_net, load_net, vis_detections

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='personpart', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='res101.yml',
                      default='cfgs/personpart_hf/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res101',
                      default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=10, type=int)
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      default=1, type=int)
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)

  # set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)
  # resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default='408', type=str)
  # log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=1, type=int)
  parser.add_argument('--debug', dest='debug',
                      help='visualization mode',
                      default=1, type=int)


  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2017_val"
      args.imdbval_name = "coco_2017_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "personpart":
      args.imdb_name = "personpart_val"
      args.imdbval_name = "personpart_val"
      # args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'ANCHOR_SCALES_HF', '[2,4,8,16]', 'MAX_NUM_GT_BOXES', '20']
      args.set_cfgs = []

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not cfg.CUDA:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = cfg.CUDA
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = cfg.SAVE_DIR + "/" + args.dataset + "/" + cfg.EXP_DIR
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, 1)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                           shuffle=False, num_workers=args.num_workers,
                                           pin_memory=True)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)


  # ship to cuda
  if cfg.CUDA:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if cfg.CUDA:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=cfg.CLASS_AGNOSTIC)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=cfg.CLASS_AGNOSTIC)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=cfg.CLASS_AGNOSTIC)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=cfg.CLASS_AGNOSTIC)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
      if value.requires_grad:
          if 'bias' in key:
              params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                          'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
          else:
              params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
      lr = lr * 0.1
      optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
      optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if cfg.CUDA:
    fasterRCNN.cuda()

  iters_per_epoch = int(train_size / 1)

  for epoch in range(args.start_epoch, cfg.TRAIN.EPOCHS + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()
    
    if epoch in cfg.TRAIN.LR_SCHEDULE:
        adjust_learning_rate(optimizer, cfg.TRAIN.GAMMA)
        lr *= cfg.TRAIN.GAMMA

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0]) #(12,3,902,600)
      im_info.data.resize_(data[1].size()).copy_(data[1]) #(12,3) [902.0000  600.0000  1.4085]
      gt_boxes.data.resize_(data[2].size()).copy_(data[2]) #(12,50,5)
      num_boxes.data.resize_(data[3].size()).copy_(data[3]) #(12,) boxes num of each image

      fasterRCNN.zero_grad()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
      rois_hf, cls_prob_hf, bbox_pred_hf, \
      rpn_loss_cls_hf, rpn_loss_bbox_hf, \
      RCNN_loss_cls_hf, RCNN_loss_bbox_hf, rois_label_hf, person_rois = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      if args.debug:
          rois_hf_origin = rois_hf.clone()  # 32*300*5 # 1*32*5
          for m in range(person_rois.size(1)):  # 32
              for n in range(rois_hf.size(1)):  # 300
                  scale_x = cfg.POOLING_SIZE_PERSON * 16 / (person_rois[0, m, 3] - person_rois[0, m, 1])
                  scale_y = cfg.POOLING_SIZE_PERSON * 16 / (person_rois[0, m, 4] - person_rois[0, m, 2])
                  rois_hf_origin[m, n, 1] = person_rois[0, m, 1] + rois_hf_origin[m, n, 1] / scale_x
                  rois_hf_origin[m, n, 2] = person_rois[0, m, 2] + rois_hf_origin[m, n, 2] / scale_y
                  rois_hf_origin[m, n, 3] = person_rois[0, m, 1] + rois_hf_origin[m, n, 3] / scale_x
                  rois_hf_origin[m, n, 4] = person_rois[0, m, 2] + rois_hf_origin[m, n, 4] / scale_y

          boxes = rois.data[:, :, 1:5]
          boxes = boxes.view(-1, 4)

          boxes_handface = rois_hf_origin.data[:, :, 1:5]
          boxes_handface = boxes_handface.view(-1, 4)
          output_dir = './debug'
          if not os.path.exists(output_dir):
              os.makedirs(output_dir)
      if args.vis:
          im = im_data.permute(0, 2, 3, 1).contiguous().view(im_data.size(2), im_data.size(3), 3)
          im2show_1s = np.copy(im)
          im2show_2s = np.copy(im)

          im2show_1s = vis_rois(im2show_1s, boxes.cpu().numpy(), rois_label.cpu().numpy(), person_flag=1)
          cv2.imwrite('./debug/' + str(step) + '_1s.png', im2show_1s)

          im2show_2s = vis_rois(im2show_2s, boxes_handface.cpu().numpy(), rois_label_hf.cpu().numpy(),
                                person_flag=0)
          cv2.imwrite('./debug/' + str(step) + '_2s.png', im2show_2s)

      # loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
      #           + rpn_loss_cls_hf.mean() + rpn_loss_bbox_hf.mean() + RCNN_loss_cls_hf.mean() + RCNN_loss_bbox_hf.mean()
      #
      # loss_temp += loss.data[0]

      # backward
      # optimizer.zero_grad()
      # loss.backward()

      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      # optimizer.step()

      if step % cfg.TRAIN.DISPLAY == 0:
        end = time.time()
        if step > 0:
          loss_temp /= cfg.TRAIN.DISPLAY

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().data[0]
          loss_rpn_box = rpn_loss_box.mean().data[0]
          loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
          loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
          
          loss_rpn_cls_hf = rpn_loss_cls_hf.mean().data[0]
          loss_rpn_box_hf = rpn_loss_bbox_hf.mean().data[0]
          loss_rcnn_cls_hf= RCNN_loss_cls_hf.mean().data[0]
          loss_rcnn_box_hf = RCNN_loss_bbox_hf.mean().data[0]

          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
          fg_cnt_hf = torch.sum(rois_label_hf.data.ne(0))
          bg_cnt_hf = rois_label_hf.data.numel() - fg_cnt_hf

        else:
          loss_rpn_cls = rpn_loss_cls.data[0]
          loss_rpn_box = rpn_loss_box.data[0]
          loss_rcnn_cls = RCNN_loss_cls.data[0]
          loss_rcnn_box = RCNN_loss_bbox.data[0]
          
          loss_rpn_cls_hf = rpn_loss_cls_hf.data[0]
          loss_rpn_box_hf = rpn_loss_bbox_hf.data[0]
          loss_rcnn_cls_hf= RCNN_loss_cls_hf.data[0]
          loss_rcnn_box_hf = RCNN_loss_bbox_hf.data[0]

          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
          fg_cnt_hf = torch.sum(rois_label_hf.data.ne(0))
          bg_cnt_hf = rois_label_hf.data.numel() - fg_cnt_hf

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
              % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), fg_hf/bg_hf=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, fg_cnt_hf, bg_cnt_hf, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, rpn_hf_cls %.4f, rpn_hf_bbox %.4f, rcnn_cls_hf: %.4f, rcnn_box_hf %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_rpn_cls_hf, loss_rpn_box_hf, loss_rcnn_cls_hf, loss_rcnn_box_hf))

        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box,
            'loss_rpn_cls_hf': loss_rpn_cls_hf,
            'loss_rpn_box_hf': loss_rpn_box_hf,
            'loss_rcnn_cls_hf': loss_rcnn_cls_hf,
            'loss_rcnn_box_hf': loss_rcnn_box_hf,
            'lr': lr
          }
          for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        loss_temp = 0
        start = time.time()

    if args.mGPUs:
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, args.checkpoint))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': fasterRCNN.module.state_dict(),
        'lr': lr,
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': cfg.CLASS_AGNOSTIC,
      }, save_name)
    else:
      save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, args.checkpoint))
      save_checkpoint({
        'session': args.session,
        'epoch': epoch + 1,
        'model': fasterRCNN.state_dict(),
        'lr': lr,
        'pooling_mode': cfg.POOLING_MODE,
        'class_agnostic': cfg.CLASS_AGNOSTIC,
      }, save_name)
    print('save model: {}'.format(save_name))

    end = time.time()
    print(end - start)
