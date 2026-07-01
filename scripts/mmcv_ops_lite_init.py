from __future__ import annotations

import torch
from torch import Tensor


def point_sample(input: Tensor, point_coords: Tensor, align_corners: bool = False, **kwargs) -> Tensor:
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    else:
        add_dim = False
    output = torch.nn.functional.grid_sample(
        input, 2.0 * point_coords - 1.0, align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def sigmoid_focal_loss(inputs: Tensor,
                       targets: Tensor,
                       gamma: float = 2.0,
                       alpha: float = 0.25,
                       weight: Tensor | None = None,
                       reduction: str = 'mean') -> Tensor:
    prob = inputs.sigmoid()
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if weight is not None:
        loss = loss * weight
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'none':
        return loss
    raise ValueError(f'Unsupported reduction: {reduction}')


def softmax_focal_loss(*args, **kwargs):
    raise RuntimeError('softmax_focal_loss requires compiled mmcv ops in this environment')


class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean') -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor, weight: Tensor | None = None) -> Tensor:
        return sigmoid_focal_loss(inputs, targets, self.gamma, self.alpha, weight, self.reduction)


class SoftmaxFocalLoss(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return softmax_focal_loss(*args, **kwargs)


def _missing_op(*args, **kwargs):
    raise RuntimeError('This mmcv op requires compiled mmcv._ext, which is unavailable in this environment')


class _MissingOp(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return _missing_op(*args, **kwargs)


CARAFE = CARAFENaive = CARAFEPack = CornerPool = DeformConv2d = DeformConv2dPack = _MissingOp
DeformRoIPool = DeformRoIPoolPack = ModulatedDeformRoIPoolPack = _MissingOp
MaskedConv2d = ModulatedDeformConv2d = ModulatedDeformConv2dPack = _MissingOp
RoIAlign = RoIPool = SyncBatchNorm = CrissCrossAttention = PSAMask = _MissingOp
SimpleRoIAlign = SAConv2d = TINShift = RoIPointPool3d = FusedBiasLeakyReLU = _MissingOp
RiRoIAlignRotated = RoIAlignRotated = QueryAndGroup = GroupAll = PointsSampler = _MissingOp
Correlation = Voxelization = DynamicScatter = RoIAwarePool3d = _MissingOp
SparseConv2d = SparseConv3d = SparseConvTranspose2d = SparseConvTranspose3d = _MissingOp
SparseInverseConv2d = SparseInverseConv3d = SubMConv2d = SubMConv3d = _MissingOp
SparseModule = SparseSequential = SparseMaxPool2d = SparseMaxPool3d = _MissingOp
SparseConvTensor = PrRoIPool = BezierAlign = MultiScaleDeformableAttention = _MissingOp

active_rotated_filter = assign_score_withk = ball_query = bbox_overlaps = bezier_align = _missing_op
bias_act = border_align = box_iou_quadri = box_iou_rotated = carafe = carafe_naive = _missing_op
chamfer_distance = contour_expand = conv2d = conv_transpose2d = convex_giou = convex_iou = _missing_op
deform_conv2d = deform_roi_pool = diff_iou_rotated_2d = diff_iou_rotated_3d = _missing_op
filter2d = filtered_lrelu = furthest_point_sample = furthest_point_sample_with_dist = _missing_op
fused_bias_leakyrelu = gather_points = get_compiler_version = get_compiling_cuda_version = _missing_op
grouping_operation = knn = masked_conv2d = min_area_polygons = modulated_deform_conv2d = _missing_op
nms = batched_nms = soft_nms = nms_match = nms_quadri = nms_rotated = _missing_op
boxes_iou3d = boxes_iou_bev = boxes_overlap_bev = nms3d = nms3d_normal = nms_bev = nms_normal_bev = _missing_op
pixel_group = points_in_boxes_all = points_in_boxes_cpu = points_in_boxes_part = _missing_op
points_in_polygons = prroi_pool = rel_roi_point_to_rel_img_point = riroi_align_rotated = _missing_op
roi_align = roi_align_rotated = roi_pool = rotated_feature_align = scatter_nd = dynamic_scatter = _missing_op
three_interpolate = three_nn = tin_shift = upfirdn2d = upsample2d = voxelization = _missing_op

Conv2d = torch.nn.Conv2d
ConvTranspose2d = torch.nn.ConvTranspose2d
Linear = torch.nn.Linear
MaxPool2d = torch.nn.MaxPool2d

__all__ = [name for name in globals() if not name.startswith('_')]
