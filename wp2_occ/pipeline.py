import json
import os

import numpy as np

from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadWP2OccTarget(object):
    """Load the Occ3D occupancy target of a frame: <occ_path>/labels.npz -> results['wp2_occ_sem']
    (200, 200, 16 uint8, [x][y][z], classes 0-17) and results['wp2_occ_mask'] (mask_camera).

    shuffle_map (str, optional): JSON {sample token: occ_path of the frame whose occupancy is
    used instead}. The shuffled control (WP2): each frame gets the occupancy of a frame from a
    DIFFERENT scene (fixed pairing, np.random.default_rng(0); built by
    thesis-bevdet/scripts/wp2/build_shuffle_map.py). Every token must be in the map.
    The keys differ from BEVAug's 'voxel_semantics'/'mask_camera' on purpose: BEVAug only flips
    those, while the model applies the full bda matrix (wp2_occ.model.occ_target_under_bda).
    """

    def __init__(self, shuffle_map=None):
        self.shuffle_map_path = shuffle_map
        self.shuffle_map = json.load(open(shuffle_map)) if shuffle_map else None

    def __call__(self, results):
        if self.shuffle_map is None:
            occ_path = results['curr']['occ_path']
        else:
            occ_path = self.shuffle_map[results['sample_idx']]
        if occ_path.startswith('./'):
            occ_path = occ_path[2:]
        d = np.load(os.path.join(occ_path, 'labels.npz'))
        sem, mask = d['semantics'], d['mask_camera']
        assert sem.shape == (200, 200, 16) and mask.shape == (200, 200, 16), (sem.shape, mask.shape)
        results['wp2_occ_sem'] = sem
        results['wp2_occ_mask'] = mask
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(shuffle_map={self.shuffle_map_path})'
