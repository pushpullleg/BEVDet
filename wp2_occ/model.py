# WP2 (thesis, representation level, reduced form): occupancy supervision of BEVDet's BEV features
# during fine-tuning, with the oracle Occ3D teacher (an idealized setting). mmdet3d/ is not edited:
# this package is loaded through `custom_imports` in configs/bevdet/wp2/*.py and needs the
# directory that holds it on sys.path (PYTHONPATH; see configs/bevdet/wp2/wp2_occ_base.py).
#
# Head (training only): BEV encoder output F (B, 256, 128, 128), layout [y][x] -> crop the
# central 100 x 100 cells (offset 14 = Occ3D +-40 m) -> transpose to [x][y] like Occ3D ->
# 2x bilinear upsample to 200 x 200 (0.4 m) -> 3x3 conv-BN-ReLU, 3x3 conv-BN-ReLU, 1x1 conv ->
# 16 heights x 18 classes. Loss: cross-entropy over the 18 Occ3D classes on camera-visible voxels
# (mask_camera), weight 1.0, added to the detection losses.
#
# BEV augmentation: the WP1 schedule trains with BEVAug (rotation +-22.5 deg, scale 0.95-1.05,
# x/y flips). The view transformer applies the 4x4 bda matrix to the frustum points, so F is in
# the AUGMENTED frame. The targets are moved into the same frame here, with the same matrix:
# every output voxel centre p_aug is mapped back to p_src = R^-1 (p_aug - t) and takes the
# Occ3D label of the voxel that contains p_src (nearest neighbour; outside the Occ3D grid ->
# ignored). BEVAug's own occupancy code (flips only, keys 'voxel_semantics'/'mask_*') is NOT
# used: the targets travel under different keys so it cannot touch them.
#
# Test time: forward_test is BEVDet's, the head is never called. The deployed model is this
# checkpoint without the 'occ_head.' keys (thesis-bevdet/scripts/wp2/strip_occ_head.py), which
# loads into configs/bevdet/bevdet-r50-cbgs.py and runs in the existing clean / PGD / transfer
# scripts unchanged.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import HOOKS, Hook

from mmdet.models import DETECTORS
from mmdet3d.models.detectors.bevdet import BEVDet

OCC_LOWER = (-40.0, -40.0, -1.0)   # Occ3D-nuScenes grid, ego frame
OCC_VOXEL = 0.4
OCC_SHAPE = (200, 200, 16)          # [x][y][z]
N_CLASSES = 18                      # 0-16 semantic, 17 = free
BEV_OFFSET = 14                     # Occ3D +-40 m = BEV cells 14..113 of 128 (0.8 m, from -51.2 m)
BEV_CROP = 100


def occ_target_under_bda(sem, mask, bda):
    """Move Occ3D targets into the BEV-augmented frame.

    sem, mask: (B, 200, 200, 16) [x][y][z] (any integer dtype); bda: (B, 4, 4) or (B, 3, 3),
    p_aug = R p + t as in LSSViewTransformer.get_lidar_coor. Returns (sem_aug long, mask_aug
    float): label/mask of the source voxel containing R^-1 (centre_aug - t); 0 mask where the
    source falls outside the grid. Identity bda reproduces the input exactly.
    """
    B = sem.shape[0]
    dev = sem.device
    X, Y, Z = OCC_SHAPE
    lower = torch.tensor(OCC_LOWER, device=dev, dtype=torch.float64)
    ix, iy, iz = torch.meshgrid(torch.arange(X, device=dev), torch.arange(Y, device=dev),
                                torch.arange(Z, device=dev), indexing='ij')
    idx = torch.stack([ix, iy, iz], -1).double()                       # (X, Y, Z, 3)
    centres = lower + (idx + 0.5) * OCC_VOXEL                           # augmented-frame centres
    bda = bda.to(device=dev, dtype=torch.float64)
    R = bda[:, :3, :3]
    t = bda[:, :3, 3] if bda.shape[-1] == 4 else torch.zeros(B, 3, device=dev, dtype=torch.float64)
    Rinv = torch.inverse(R)
    p = centres.view(1, -1, 3) - t.view(B, 1, 3)
    src = torch.einsum('bij,bnj->bni', Rinv, p)                         # (B, XYZ, 3)
    sidx = torch.floor((src - lower) / OCC_VOXEL).long()
    shape = torch.tensor([X, Y, Z], device=dev)
    valid = ((sidx >= 0) & (sidx < shape)).all(-1)                      # (B, XYZ)
    sidx = torch.where(valid.unsqueeze(-1), sidx, torch.zeros_like(sidx))
    flat = (sidx[..., 0] * Y + sidx[..., 1]) * Z + sidx[..., 2]         # (B, XYZ)
    sem_aug = torch.gather(sem.reshape(B, -1).long(), 1, flat)
    mask_aug = torch.gather(mask.reshape(B, -1).float(), 1, flat) * valid.float()
    return sem_aug.view(B, X, Y, Z), mask_aug.view(B, X, Y, Z)


class WP2OccHead(nn.Module):
    """BEV feature (B, C, 128, 128) [y][x] -> occupancy logits (B, 18, 200, 200, 16) [x][y][z]."""

    def __init__(self, in_channels=256, mid_channels=128, num_classes=N_CLASSES, num_z=16):
        super().__init__()
        self.num_classes, self.num_z = num_classes, num_z
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(mid_channels, num_z * num_classes, 1)

    def forward(self, bev):
        x = bev[:, :, BEV_OFFSET:BEV_OFFSET + BEV_CROP, BEV_OFFSET:BEV_OFFSET + BEV_CROP]  # [y][x]
        x = x.transpose(-1, -2)                                                          # [x][y]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)      # 200 x 200
        x = self.cls(self.conv2(self.conv1(x)))                                          # (B, Z*C, X, Y)
        B, _, X, Y = x.shape
        x = x.view(B, self.num_z, self.num_classes, X, Y)
        return x.permute(0, 2, 3, 4, 1).contiguous()                                    # (B, C, X, Y, Z)


@DETECTORS.register_module()
class BEVDetWP2Occ(BEVDet):
    """BEVDet + training-only occupancy head on the BEV encoder output (same features as the
    detection head). occ_head: dict(in_channels, mid_channels); loss_occ_weight: 1.0."""

    def __init__(self, occ_head=None, loss_occ_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.occ_head = WP2OccHead(**(occ_head or {}))
        self.loss_occ_weight = float(loss_occ_weight)

    def loss_occ(self, bev, occ_sem, occ_mask, bda):
        logits = self.occ_head(bev)                                       # (B, 18, X, Y, Z)
        sem, mask = occ_target_under_bda(occ_sem, occ_mask, bda)
        ce = F.cross_entropy(logits.float(), sem, reduction='none')       # (B, X, Y, Z)
        loss = (ce * mask).sum() / mask.sum().clamp(min=1.0)
        return self.loss_occ_weight * loss

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None, gt_labels_3d=None,
                      gt_labels=None, gt_bboxes=None, img_inputs=None, proposals=None,
                      gt_bboxes_ignore=None, wp2_occ_sem=None, wp2_occ_mask=None, **kwargs):
        img_feats, _, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses.update(self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas,
                                             gt_bboxes_ignore))
        losses['loss_occ'] = self.loss_occ(img_feats[0], wp2_occ_sem, wp2_occ_mask, img_inputs[6])
        return losses


@HOOKS.register_module()
class WP2SeedCheckHook(Hook):
    """Fail fast if the --seed passed to tools/train.py differs from the seed the config is for."""

    def __init__(self, seed):
        self.seed = seed

    def before_run(self, runner):
        got = (runner.meta or {}).get('seed')
        assert got == self.seed, f'config is for seed {self.seed}, tools/train.py got --seed {got}'
