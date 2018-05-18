import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn_ms.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn_ms.proposal_target_layer_cascade import _ProposalTargetLayer

from model.rpn_ms.bbox_transform import clip_boxes, clip_boxes_batch
from model.rpn_ms.bbox_transform import bbox_transform_inv, bbox_transform_batch
from model.nms.nms_wrapper import nms

import time
import pdb
import numpy as np
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, vis_rois


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.RCNN_loss_cls_handface = 0
        self.RCNN_loss_bbox_handface = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model, person_flag=1)
        self.RCNN_rpn_handface = _RPN(self.dout_base_model, person_flag=0)

        self.RCNN_proposal_target = _ProposalTargetLayer(4)
        self.RCNN_proposal_target_handface = _ProposalTargetLayer(3)

        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.RCNN_roi_pool_person = _RoIPooling(cfg.POOLING_SIZE_PERSON, cfg.POOLING_SIZE_PERSON, 1.0 / 16.0)
        self.RCNN_roi_align_person = RoIAlignAvg(cfg.POOLING_SIZE_PERSON, cfg.POOLING_SIZE_PERSON, 1.0 / 16.0)

        self.RCNN_roi_pool_hf = _RoIPooling(cfg.POOLING_SIZE_HF, cfg.POOLING_SIZE_HF, 1.0 / 4.0)
        self.RCNN_roi_align_hf = RoIAlignAvg(cfg.POOLING_SIZE_HF, cfg.POOLING_SIZE_HF, 1.0 / 4.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_roi_feat_ds, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        
        normal_init(self.RCNN_rpn_handface.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn_handface.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn_handface.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_handface, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_handface, 0, 0.001, cfg.TRAIN.TRUNCATED)

        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top_hf, 0, 0.01, cfg.TRAIN.TRUNCATED)


    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p4 = self.RCNN_smooth1(p4)
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p3 = self.RCNN_smooth2(p3)
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p2 = self.RCNN_smooth3(p2)

        base_feat = c5
        fuse_feat = p2
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, person_flag=1)
        # if it is training phrase, then use ground truth but bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, person_flag=1)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)
        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        if self.class_agnostic or self.training:
            box_deltas = bbox_pred.data
        else:
            box_deltas = bbox_pred[:, :, 4:].contiguous().data

        if self.class_agnostic:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(batch_size, -1, 4)
        else:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(batch_size, -1, 4 * 4)

        pred_boxes_person = bbox_transform_inv(boxes, box_deltas, batch_size)
        pred_boxes_person = clip_boxes(pred_boxes_person, im_info, batch_size)
        if self.training:
            pre_nms_topN = cfg.TRAIN.BATCH_SIZE
        else:
            pre_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N

        post_nms_topN = cfg.PERSON_NMS

        scores_person = scores[:, :, 1].contiguous()
        proposals_person = pred_boxes_person

        if self.training:
            nms_thresh = 0.7
            thresh = 0.0
            num_proposal = post_nms_topN
            output = scores.new(batch_size, num_proposal, 5).zero_()
        else:
            nms_thresh = 0.3
            thresh = 0.3

        for i in range(batch_size):
            proposals_single = proposals_person[i]
            scores_single = scores_person[i]
            inds = torch.nonzero(scores_single >= thresh)
            scores_single = scores_single[inds[:, 0]]
            proposals_single = proposals_single[inds[:, 0], :]
            _, order_single = torch.sort(scores_single, 0, True)

            if pre_nms_topN > 0 and pre_nms_topN < scores_person.numel():
                order_single = order_single[:pre_nms_topN]
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)
            if self.training:
                if post_nms_topN > 0:
                    keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            # padding 0 at the end.
            if self.training:
                num_proposal = post_nms_topN
            else:
                num_proposal = proposals_single.size(0)
                output = scores.new(batch_size, num_proposal, 5).zero_()
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposals_single
        person_rois = Variable(output)

        batch_size_person = batch_size * num_proposal
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat_person = self.RCNN_roi_align_person(base_feat, person_rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_person = self.RCNN_roi_pool_person(base_feat, person_rois.view(-1, 5))
        # prepare data for handface detection
        handface_boxes, handface_num_boxes = self.generate_handface_boxes(person_rois, gt_boxes)
        if self.training:
            handface_boxes = handface_boxes.view(-1, gt_boxes.size(1), gt_boxes.size(2))
            handface_num_boxes = handface_num_boxes.view(-1)
            person_info = torch.ones(batch_size_person, 3).type_as(im_info)
            for i in range(person_info.size(0)):
                person_info[i][0:2] = cfg.POOLING_SIZE_PERSON * cfg.FEAT_STRIDE[0]
        else:
            person_info = torch.ones(cfg.TEST.RPN_POST_NMS_TOP_N, 3).type_as(im_info)
            for i in range(person_info.size(0)):
                person_info[i][0:2] = cfg.POOLING_SIZE_PERSON * cfg.FEAT_STRIDE[0]

        rois_handface, rpn_loss_cls_handface, rpn_loss_bbox_handface = self.RCNN_rpn_handface(pooled_feat_person,
                                                                                              person_info,
                                                                                              handface_boxes,
                                                                                              handface_num_boxes,
                                                                                              person_flag=0)
        # if it is training phrase, then use ground truth but bboxes for refining
        if self.training:
            roi_data_handface = self.RCNN_proposal_target_handface(rois_handface, handface_boxes, handface_num_boxes, person_flag=0)
            rois_handface, rois_label_handface, rois_target_handface, rois_inside_ws_handface, rois_outside_ws_handface = roi_data_handface
            rois_label_handface = Variable(rois_label_handface.view(-1).long())
            rois_target_handface = Variable(rois_target_handface.view(-1, rois_target_handface.size(2)))
            rois_inside_ws_handface = Variable(rois_inside_ws_handface.view(-1, rois_inside_ws_handface.size(2)))
            rois_outside_ws_handface = Variable(rois_outside_ws_handface.view(-1, rois_outside_ws_handface.size(2)))
        else:
            rois_label_handface = None
            rois_target_handface = None
            rois_inside_ws_handface = None
            rois_outside_ws_handface = None
            rpn_loss_cls_handface = 0
            rpn_loss_bbox_handface = 0

        rois_handface = Variable(rois_handface)
        if self.training:
            rois_handface_cv, rois_target_handface_cv = self.convert_handface_boxes(rois_handface, rois_target_handface, person_rois)
            rois_handface = rois_handface_cv.view(rois_handface.size(0), rois_handface.size(1), rois_handface.size(2))
            rois_target_handface = rois_target_handface_cv.view(rois_target_handface.size(0), rois_target_handface.size(1))
        else:
            rois_handface_cv, rois_target_handface_cv = self.convert_handface_boxes(rois_handface, rois_target_handface, person_rois)
            rois_handface = rois_handface_cv.view(rois_handface.size(0), rois_handface.size(1), rois_handface.size(2))

        # do roi pooling based on predicted roi
        if cfg.POOLING_MODE == 'align':
            pooled_feat_handface = self.RCNN_roi_align_hf(fuse_feat, rois_handface.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_handface = self.RCNN_roi_pool_hf(fuse_feat, rois_handface.view(-1, 5))
        # feed pooled features to top model
        
        # pooled_feat_handface = self.RCNN_roi_feat_ds(pooled_feat_handface)
        pooled_feat_handface = self._head_to_tail_handface(pooled_feat_handface)
        # compute bbox offset
        bbox_pred_handface = self.RCNN_bbox_pred_handface(pooled_feat_handface)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view_handface = bbox_pred_handface.view(bbox_pred_handface.size(0),
                                                              int(bbox_pred_handface.size(1) / 4), 4)
            bbox_pred_select_handface = torch.gather(bbox_pred_view_handface, 1,
                                                     rois_label_handface.view(rois_label_handface.size(0), 1, 1).expand(
                                                         rois_label_handface.size(0), 1, 4))
            bbox_pred_handface = bbox_pred_select_handface.squeeze(1)

        # compute object classification probability
        cls_score_handface = self.RCNN_cls_score_handface(pooled_feat_handface)
        cls_prob_handface = F.softmax(cls_score_handface, 1)

        RCNN_loss_cls_handface = 0
        RCNN_loss_bbox_handface = 0

        if self.training:
            # classification loss
            RCNN_loss_cls_handface = F.cross_entropy(cls_score_handface, rois_label_handface)
            # bounding box regression L1 loss
            RCNN_loss_bbox_handface = _smooth_l1_loss(bbox_pred_handface, rois_target_handface, rois_inside_ws_handface,
                                                      rois_outside_ws_handface)
        cls_prob_handface = cls_prob_handface.view(batch_size_person, rois_handface.size(1), -1)
        bbox_pred_handface = bbox_pred_handface.view(batch_size_person, rois_handface.size(1), -1)
        # rois_target_handface = rois_target_handface.view(batch_size_person, rois_handface.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
               rois_handface, cls_prob_handface, bbox_pred_handface, rpn_loss_cls_handface, rpn_loss_bbox_handface, \
               RCNN_loss_cls_handface, RCNN_loss_bbox_handface, rois_label_handface, person_rois

    def generate_handface_boxes(self, person_rois, gt_boxes):
        # 3*128*5=i*j*5  3*20*5=i*k*5 hf_boxes = i*j*k*5
        if self.training:
            hf_boxes = gt_boxes.new(person_rois.size(0), person_rois.size(1), gt_boxes.size(1),
                                    gt_boxes.size(2)).zero_()
            hf_num_boxes = torch.zeros(person_rois.size(0), person_rois.size(1))
            for i in range(person_rois.size(0)):
                for j in range(person_rois.size(1)):
                    num = 0
                    for k in range(gt_boxes.size(1)):
                        if gt_boxes[i][k][4] == 2 or gt_boxes[i][k][4] == 3:
                            x1 = np.maximum(person_rois.data[i, j, 1], gt_boxes[i, k, 0])
                            y1 = np.maximum(person_rois.data[i, j, 2], gt_boxes[i, k, 1])
                            x2 = np.minimum(person_rois.data[i, j, 3], gt_boxes[i, k, 2])
                            y2 = np.minimum(person_rois.data[i, j, 4], gt_boxes[i, k, 3])
                            w = np.maximum(0, x2 - x1)
                            h = np.maximum(0, y2 - y1)
                            gt_area = (gt_boxes[i, k, 3] - gt_boxes[i, k, 1]) * (gt_boxes[i, k, 2] - gt_boxes[i, k, 0])
                            if (w * h) / gt_area >= cfg.RATIO:
                                num = num + 1
                                # ensure that the handface boxes are in the person boxes
                                xx1 = person_rois.data[i, j, 1] if gt_boxes[i, k, 0] < person_rois.data[i, j, 1] else \
                                gt_boxes[i, k, 0]
                                yy1 = person_rois.data[i, j, 2] if gt_boxes[i, k, 1] < person_rois.data[i, j, 2] else \
                                gt_boxes[i, k, 1]
                                xx2 = person_rois.data[i, j, 3] if gt_boxes[i, k, 2] > person_rois.data[i, j, 3] else \
                                gt_boxes[i, k, 2]
                                yy2 = person_rois.data[i, j, 4] if gt_boxes[i, k, 3] > person_rois.data[i, j, 4] else \
                                gt_boxes[i, k, 3]
                                # scale of person_size-->14*16
                                scale_x = cfg.POOLING_SIZE_PERSON * 16 / (
                                            person_rois.data[i, j, 3] - person_rois.data[i, j, 1])
                                scale_y = cfg.POOLING_SIZE_PERSON * 16 / (
                                            person_rois.data[i, j, 4] - person_rois.data[i, j, 2])
                                # calculate the handface boxes positon in the person box and resize it according to the resize ratio of person boxes
                                hf_boxes[i, j, num - 1, 0] = scale_x * (xx1 - person_rois.data[i, j, 1])  # x1
                                hf_boxes[i, j, num - 1, 1] = scale_y * (yy1 - person_rois.data[i, j, 2])  # y1
                                hf_boxes[i, j, num - 1, 2] = scale_x * (xx2 - person_rois.data[i, j, 1])  # x2
                                hf_boxes[i, j, num - 1, 3] = scale_y * (yy2 - person_rois.data[i, j, 2])  # y2
                                hf_boxes[i, j, num - 1, 4] = gt_boxes[i, k, 4]  # class(hand/face)
                    hf_num_boxes[i, j] = num
        else:
            hf_boxes = None
            hf_num_boxes = None
        return hf_boxes, hf_num_boxes

    def convert_handface_boxes(self, rois_hf, rois_target_hf, person_rois):
        # ((12L, 64L, 5L), (768L,), (768L, 4L), (2L, 6L, 5L))
        # ((2L, 6L, 64L, 5L), (768L,), (2L, 6L, 64L, 4L), (2L, 6L, 5L))
        if self.training:
            rois_hf = rois_hf.view(person_rois.size(0), person_rois.size(1), cfg.TRAIN.BATCH_SIZE_HF, 5)
            rois_target_hf = rois_target_hf.view(person_rois.size(0), person_rois.size(1), cfg.TRAIN.BATCH_SIZE_HF, 4)
            rois_hf_origin = rois_hf.clone().zero_()
            rois_target_hf_origin = rois_target_hf.clone().zero_()
            for m in range(person_rois.size(0)):
                for n in range(person_rois.size(1)):
                    scale_x = cfg.POOLING_SIZE_PERSON * 16 / (person_rois[m, n, 3] - person_rois[m, n, 1])
                    scale_y = cfg.POOLING_SIZE_PERSON * 16 / (person_rois[m, n, 4] - person_rois[m, n, 2])
                    rois_hf_origin[m, n, :, 1] = person_rois[m, n, 1] + rois_hf[m, n, :, 1] / scale_x
                    rois_hf_origin[m, n, :, 2] = person_rois[m, n, 2] + rois_hf[m, n, :, 2] / scale_y
                    rois_hf_origin[m, n, :, 3] = person_rois[m, n, 1] + rois_hf[m, n, :, 3] / scale_x
                    rois_hf_origin[m, n, :, 4] = person_rois[m, n, 2] + rois_hf[m, n, :, 4] / scale_y
                    # dx dy dw dh
                    rois_target_hf_origin[m, n, :, 0] = rois_target_hf[m, n, :, 0] / scale_x
                    rois_target_hf_origin[m, n, :, 1] = rois_target_hf[m, n, :, 1] / scale_y
                    rois_target_hf_origin[m, n, :, 2] = rois_target_hf[m, n, :, 2] / scale_x
                    rois_target_hf_origin[m, n, :, 3] = rois_target_hf[m, n, :, 3] / scale_y
        else:
            rois_hf = rois_hf.view(person_rois.size(0), person_rois.size(1), rois_hf.size(1), 5)
            rois_hf_origin = rois_hf.clone().zero_()
            for m in range(person_rois.size(0)):
                for n in range(person_rois.size(1)):
                    scale_x = cfg.POOLING_SIZE_PERSON * 16 / (person_rois[m, n, 3] - person_rois[m, n, 1])
                    scale_y = cfg.POOLING_SIZE_PERSON * 16 / (person_rois[m, n, 4] - person_rois[m, n, 2])
                    rois_hf_origin[m, n, :, 1] = person_rois[m, n, 1] + rois_hf[m, n, :, 1] / scale_x
                    rois_hf_origin[m, n, :, 2] = person_rois[m, n, 2] + rois_hf[m, n, :, 2] / scale_y
                    rois_hf_origin[m, n, :, 3] = person_rois[m, n, 1] + rois_hf[m, n, :, 3] / scale_x
                    rois_hf_origin[m, n, :, 4] = person_rois[m, n, 2] + rois_hf[m, n, :, 4] / scale_y
            rois_target_hf_origin = None
        return rois_hf_origin, rois_target_hf_origin






