import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import ModuleList, BaseModule, auto_fp16, force_fp32
from mmdet.models.backbones.resnet import Bottleneck
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from mmdet.models.losses import accuracy

class BasicResBlock(BaseModule):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(BasicResBlock, self).__init__(init_cfg)

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@HEADS.register_module()
class DoubleConvFCBBoxHead(BBoxHead):
    r"""Bbox head used in Double-Head R-CNN

    .. code-block:: none

                                          /-> cls
                      /-> shared convs ->
                                          \-> reg
        roi features
                                          /-> cls
                      \-> shared fc    ->
                                          \-> reg
    """  # noqa: W605

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=256,
                 fc_out_channels_cls = 512,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=dict(
                     type='Normal',
                     override=[
                         dict(type='Normal', name='fc_cls', std=0.01),
                         dict(type='Normal', name='fc_reg', std=0.001),
                         dict(type='Normal', name='fc_cls_cls', std=0.01),
                         dict(
                             type='Xavier',
                             name='fc_branch',
                             distribution='uniform'),
                         dict(
                             type='Xavier',
                             name='fc_branch_cls',
                             distribution='uniform')
                     ]),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHead, self).__init__(init_cfg=init_cfg, **kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.fc_out_channels_cls = fc_out_channels_cls
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels,
                                       self.conv_out_channels)
        self.res_block_cls = BasicResBlock(self.in_channels ,
                                       self.conv_out_channels)

        # add conv heads
        self.conv_branch = self._add_conv_branch(self.num_convs)
        # add fc heads
        self.fc_branch = self._add_fc_branch()
        # add conv cls heads
        self.conv_branch_cls = self._add_conv_branch(self.num_convs-1)
        self.conv_branch_cls.append(BasicResBlock(self.conv_out_channels ,
                                       self.conv_out_channels // 4)) #1024->256
        # add fc cls heads
        self.fc_branch_cls = self._add_fc_branch_cls()
        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes

        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)#256->4
        self.fc_cls_cls = nn.Linear(self.fc_out_channels_cls, self.num_classes_cls + 1)#用于类别分类,512->56
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_branch(self,num_convs):
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = ModuleList()
        for i in range(num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,#bottleneck会乘expansion，使得输出通道数增大
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers."""
        branch_fcs = ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area *2 if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def _add_fc_branch_cls(self):
        """Add the fc branch which consists of a sequential of fc layers."""
        branch_fcs = ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.conv_out_channels//4 if i == 0 else self.fc_out_channels_cls)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels_cls))
        return branch_fcs

    @auto_fp16()
    def forward(self, x_cls, x_reg):
        #conv and fc cls head
        x_conv_cls = self.res_block_cls(x_cls)#256 -> 1024
        for conv in self.conv_branch_cls:
            x_conv_cls = conv(x_conv_cls)#1024 -> 256

        if self.with_avg_pool:
            x_fc_cls = self.avg_pool(x_conv_cls)
            x_fc_cls = x_fc_cls.view(x_fc_cls.size(0),-1)
        else:
            x_fc_cls = x_conv_cls.view(x_conv_cls.size(0),-1)

        for fc in self.fc_branch_cls:
            x_fc_cls = self.relu(fc(x_fc_cls))#256->512
        cls_cls_score = self.fc_cls_cls(x_fc_cls)

        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)#1024
        bbox_pred = self.fc_reg(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)#256*roi_feat_size
        x_conv_cls = x_conv_cls.view(x_conv_cls.size(0),-1)#1024
        #print("x_fc {} x_conv_cls {}".format(x_fc.size(),x_conv_cls.size()))
        x_fc = torch.cat((x_fc, x_conv_cls),dim=1) #1024+256*roi_feat_size

        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        cls_score = self.fc_cls(x_fc)

        return cls_score, bbox_pred, cls_cls_score


    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels,pos_gt_labels_cls, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        labels_cls = pos_bboxes.new_full((num_samples,),
                                     self.num_classes_cls,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            labels_cls[:num_pos] = pos_gt_labels_cls
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, labels_cls, label_weights, bbox_targets, bbox_weights
    def get_targets(self,
                    sampling_results,
                    sampling_results_cls,
                    gt_bboxes,
                    gt_labels,
                    gt_labels_cls,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_gt_labels_list_cls = [res.pos_gt_labels for res in sampling_results_cls]
        labels, labels_cls, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_gt_labels_list_cls,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            labels_cls = torch.cat(labels_cls,0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, labels_cls,label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'cls_cls_score'))
    def loss(self,
             cls_score,
             bbox_pred,
             cls_cls_score,
             rois,
             labels,
             labels_cls,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                #loss_cls_是一个tensor，对于doubleRcnn，用的是CrossEntropy
                loss_cls_blur_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if cls_cls_score.numel()>0:
                    loss_cls_cls_ = self.loss_cls(
                        cls_cls_score,
                        labels_cls,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                    #print("cls_cls_score: ", cls_cls_score)
                else:
                    loss_cls_cls_ = None
                # -------两个分类loss在此处相加----------
                loss_cls_ = loss_cls_blur_ + 0.3*loss_cls_cls_
                #print("loss blur : ",loss_cls_blur_.item(),"loss cls : ",loss_cls_cls_.item())

                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            #print('bbox head num_classes: ', self.num_classes)
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
