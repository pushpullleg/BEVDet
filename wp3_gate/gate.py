# WP3 training-free occupancy gate (thesis). mmdet3d/ is not edited: this package is loaded
# through `custom_imports` in configs/bevdet/wp3/*.py and needs $BEVDET_ROOT on sys.path.
#
#   F' = F * (alpha + (1 - alpha) * P),  the same P for all channels.
#
# P is a 0/1 BEV prior (B, 128, 128), layout [y][x] like the BEV feature (voxel_pooling_v2 ranks
# y*X + x) and the CenterPoint heatmap. It is NOT computed from images: LoadBEVPrior reads it from
# disk by sample token and passes it in img_metas ('bev_prior_real', 'bev_prior_shuffled').
# The gate sits inside extract_img_feat, so every caller of extract_feat / forward_test
# (tools/test.py and attacks/eval_adversarial.py) runs the gated model, and attack gradients flow
# through the gate.
import numpy as np
import torch

from mmdet.models import DETECTORS
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.models.detectors.bevdet import BEVDet

GATE_DEFAULTS = dict(enable=False, location='after_vt', alpha=1.0, prior='real')


@DETECTORS.register_module()
class BEVDetGate(BEVDet):
    """BEVDet with an optional occupancy gate.

    gate (dict):
        enable (bool): default False; when False the model is BEVDet unchanged.
        location (str): 'after_vt'  = view-transformer output, before the BEV encoder;
                        'after_enc' = BEV encoder neck output, before the head.
        alpha (float): pass-through factor in [0, 1]; 1.0 = gate is the identity.
        prior (str): 'real' or 'shuffled' (prior of a frame from a different scene).
    """

    def __init__(self, gate=None, **kwargs):
        super(BEVDetGate, self).__init__(**kwargs)
        cfg = dict(GATE_DEFAULTS)
        cfg.update(gate or {})
        assert set(cfg) == set(GATE_DEFAULTS), f'unknown gate keys: {set(cfg) - set(GATE_DEFAULTS)}'
        assert cfg['location'] in ('after_vt', 'after_enc'), cfg['location']
        assert cfg['prior'] in ('real', 'shuffled'), cfg['prior']
        assert 0.0 <= float(cfg['alpha']) <= 1.0, cfg['alpha']
        cfg['alpha'] = float(cfg['alpha'])
        self.gate_cfg = cfg

    def gate(self, x, img_metas):
        key = 'bev_prior_' + self.gate_cfg['prior']
        assert len(img_metas) == x.shape[0], (len(img_metas), x.shape)
        prior = torch.stack([torch.as_tensor(np.asarray(m[key])) for m in img_metas])
        prior = prior.to(device=x.device, dtype=x.dtype).unsqueeze(1)   # (B, 1, Y, X)
        assert prior.shape[-2:] == x.shape[-2:], (prior.shape, x.shape)
        alpha = self.gate_cfg['alpha']
        return x * (alpha + (1.0 - alpha) * prior)

    def extract_img_feat(self, img, img_metas, **kwargs):
        """BEVDet.extract_img_feat with the gate inserted at the configured location."""
        on = self.gate_cfg['enable']
        loc = self.gate_cfg['location']
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        if on and loc == 'after_vt':
            x = self.gate(x, img_metas)
        x = self.bev_encoder(x)
        if on and loc == 'after_enc':
            x = self.gate(x, img_metas)
        return [x], depth


@PIPELINES.register_module()
class LoadBEVPrior(object):
    """Load precomputed BEV priors by sample token: <prior_root>/<variant>/<token>.npy,
    uint8 (128, 128), [y][x]. Puts them in results['bev_prior_<variant>'] (collect them with
    Collect3D meta_keys)."""

    def __init__(self, prior_root, variants=('real', 'shuffled')):
        self.prior_root = prior_root
        self.variants = tuple(variants)

    def __call__(self, results):
        token = results['sample_idx']
        for v in self.variants:
            p = np.load(f'{self.prior_root}/{v}/{token}.npy')
            assert p.shape == (128, 128) and p.dtype == np.uint8, (p.shape, p.dtype)
            results['bev_prior_' + v] = p
        return results
