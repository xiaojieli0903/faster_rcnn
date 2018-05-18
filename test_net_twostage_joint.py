# --------------------------------------------------------
# Tensorflow Faster R-CNN
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
import torch
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms

from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_gts

from model.faster_rcnn_2s_strong.resnet_twostage import resnet
import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


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
                        default='cfgs/personpart_2s/res50_ms_joint.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=1, type=int)
    parser.add_argument('--test_mode', dest='test_mode',
                        help='visualization mode: all or 1s or 2s',
                        default='all', type=str)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not cfg.CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2017_train"
        args.imdbval_name = "coco_2017_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "personpart":
        args.imdb_name = "personpart_train"
        args.imdbval_name = "personpart_val"
        # args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'ANCHOR_SCALES_HF',
        #                  '[2,4,8,16]', 'MAX_NUM_GT_BOXES', '20']

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))
    print(cfg.LOAD_DIR, args.net, args.dataset)
    input_dir = cfg.SAVE_DIR + "/" + args.dataset + "/" + cfg.EXP_DIR + "/" + cfg.RESUME_CHECKPOINT
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, cfg.CHECKPOINT))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=cfg.CLASS_AGNOSTIC)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=cfg.CLASS_AGNOSTIC)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=cfg.CLASS_AGNOSTIC)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=cfg.CLASS_AGNOSTIC)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
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
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    if cfg.CUDA:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.00
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
        rois_hf, cls_prob_hf, bbox_pred_hf, \
        rpn_loss_cls_hf, rpn_loss_bbox_hf, \
        RCNN_loss_cls_hf, RCNN_loss_bbox_hf, rois_label_hf, person_rois = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        rois_hf_origin = rois_hf.clone() # 32*300*5 # 1*32*5
        # for m in range(person_rois.size(1)): # 32
        #     for n in range(rois_hf.size(1)): # 300
        #         scale_x = cfg.POOLING_SIZE_PERSON * cfg.FEAT_STRIDE[1] / (person_rois[0, m, 3] - person_rois[0, m, 1])
        #         scale_y = cfg.POOLING_SIZE_PERSON * cfg.FEAT_STRIDE[1] / (person_rois[0, m, 4] - person_rois[0, m, 2])
        #         # print(rois[0, m, :], rois_hf_origin[m, n, :])
        #         rois_hf_origin[m, n, 1] = person_rois[0, m, 1] + rois_hf_origin[m, n, 1] / scale_x
        #         rois_hf_origin[m, n, 2] = person_rois[0, m, 2] + rois_hf_origin[m, n, 2] / scale_y
        #         rois_hf_origin[m, n, 3] = person_rois[0, m, 1] + rois_hf_origin[m, n, 3] / scale_x
        #         rois_hf_origin[m, n, 4] = person_rois[0, m, 2] + rois_hf_origin[m, n, 4] / scale_y
        #
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        scores_hf = cls_prob_hf.data.view(1,-1,cls_prob_hf.size(2))
        boxes_hf = rois_hf_origin.data.contiguous().view(1, -1, rois_hf_origin.size(2))[:, :, 1:5]
        bbox_pred_hf.contiguous().view(1, -1, bbox_pred_hf.size(2))
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            box_deltas_hf = bbox_pred_hf.data

            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if cfg.CLASS_AGNOSTIC:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * 4)

                if cfg.CLASS_AGNOSTIC:
                    box_deltas_hf = box_deltas_hf.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas_hf = box_deltas_hf.view(1, -1, 4)
                else:
                    box_deltas_hf = box_deltas_hf.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas_hf = box_deltas_hf.view(1, -1, 4 * 3)

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

            pred_boxes_hf = bbox_transform_inv(boxes_hf, box_deltas_hf, 1)
            pred_boxes_hf = clip_boxes(pred_boxes_hf, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
            pred_boxes_hf = np.tile(boxes_hf, (1, scores_hf.shape[1]))

        pred_boxes /= data[1][0][2]
        pred_boxes_hf /= data[1][0][2]

        gt_boxes /= data[1][0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        scores_hf = scores_hf.squeeze()
        pred_boxes_hf = pred_boxes_hf.squeeze()

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        vis_thresh = 0.0
        # person_thresh = 0.3
        # hf_thresh = 0.3
        # hf_thresh_2s = 0.6
        thresh = cfg.TEST.PERSON_THRESH
        hf_thresh = cfg.TEST.HF_THRESH_S1
        hf_thresh_2s = cfg.TEST.HF_THRESH_S2
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            if args.test_mode == 'all':
                if j == 1:
                    inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if cfg.CLASS_AGNOSTIC:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_dets, cfg.TEST.NMS)

                        cls_dets = cls_dets[keep.view(-1).long()]
                        if vis:
                            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), vis_thresh)
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    else:
                        all_boxes[j][i] = empty_array
                else:
                    inds = torch.nonzero(scores[:, j] > hf_thresh).view(-1)
                    inds_hf = torch.nonzero(scores_hf[:, (j-1)] > hf_thresh_2s).view(-1)
                    # if there is det
                    if inds.numel() > 0 and inds_hf.numel() > 0:
                        cls_scores = torch.cat((scores[:, j][inds], scores_hf[:, (j-1)][inds_hf]), 0)
                        _, order = torch.sort(cls_scores, 0, True)
                        if cfg.CLASS_AGNOSTIC:
                            cls_boxes = torch.cat((pred_boxes[inds, :], pred_boxes_hf[inds_hf, :]), 0)
                        else:
                            cls_boxes = torch.cat((pred_boxes[inds][:, j * 4:(j + 1) * 4],
                                                   pred_boxes_hf[inds_hf][:, (j-1) * 4:((j-1) + 1) * 4]), 0)
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_dets, cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if vis:
                            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), vis_thresh)
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    elif inds.numel() > 0 and inds_hf.numel() == 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if cfg.CLASS_AGNOSTIC:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_dets, cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if vis:
                            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), vis_thresh)
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    elif inds.numel() == 0 and inds_hf.numel() > 0:
                        cls_scores = scores_hf[:, j-1][inds_hf]
                        _, order = torch.sort(cls_scores, 0, True)
                        if cfg.CLASS_AGNOSTIC:
                            cls_boxes = pred_boxes_hf[inds_hf, :]
                        else:
                            cls_boxes = pred_boxes_hf[inds_hf][:, (j-1) * 4:((j-1) + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_dets, cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if vis:
                            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), vis_thresh)
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    else:
                        all_boxes[j][i] = empty_array
            elif args.test_mode == '1s':
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if cfg.CLASS_AGNOSTIC:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)

                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), vis_thresh)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array
            elif args.test_mode == '2s':
                if j == 1:
                    inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if cfg.CLASS_AGNOSTIC:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_dets, cfg.TEST.NMS)

                        cls_dets = cls_dets[keep.view(-1).long()]
                        if vis:
                            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), vis_thresh)
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    else:
                        all_boxes[j][i] = empty_array
                else:
                    inds_hf = torch.nonzero(scores_hf[:, (j-1)] > hf_thresh_2s).view(-1)
                    if inds_hf.numel() > 0:
                        cls_scores = scores_hf[:, j-1][inds_hf]
                        _, order = torch.sort(cls_scores, 0, True)
                        if cfg.CLASS_AGNOSTIC:
                            cls_boxes = pred_boxes_hf[inds_hf, :]
                        else:
                            cls_boxes = pred_boxes_hf[inds_hf][:, (j-1) * 4:((j-1) + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_dets, cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        if vis:
                            im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), vis_thresh)
                        all_boxes[j][i] = cls_dets.cpu().numpy()
                    else:
                        all_boxes[j][i] = empty_array

        # im2show = vis_gts(im2show, gt_boxes.cpu().data.numpy(), num_boxes)

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()
        if args.test_mode == 'all':
            output_dir = input_dir + '/' + cfg.CHECKPOINT + '/output_all'
        if args.test_mode == '1s':
            output_dir = input_dir + '/' + cfg.CHECKPOINT + '/output_1s'
        if args.test_mode == '2s':
            output_dir = input_dir + '/' + cfg.CHECKPOINT + '/output_2s'

        det_file = os.path.join(output_dir, 'detections.pkl')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if vis:
            cv2.imwrite(output_dir + '/' + str(i) + '.png', im2show)
            # pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
