import ezc3d
from IPython import embed

data = ezc3d.c3d("./scripts/example.c3d")
embed()

assert data['header']['points']['frame_rate'] == 30

data['data']['points'] = data['data']['points'][:, :, ::3]
data['header']['points']['last_frame'] = data['header']['points']['last_frame'] // 3

data['data']['meta_points']['camera_masks'] = data['data']['meta_points']['camera_masks'][:, :, ::3]
data['data']['meta_points']['residuals'] = data['data']['meta_points']['residuals'][:, :, ::3]
