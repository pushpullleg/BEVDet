# BEVDet Visualization: First-Principles Walkthrough

This guide explains how BEVDet visualization works from first principles, using the exact code path in this repository.

## 1) What problem is BEVDet solving?

BEVDet takes multi-camera observations and predicts 3D objects around the ego vehicle.

Core idea:
1. Cameras see the world in perspective view.
2. Driving decisions are easier in top-down coordinates.
3. BEVDet converts image features into BEV (Bird's-Eye View) features.
4. A 3D detection head predicts boxes (position, size, yaw, score, class).
5. Visualization code writes points and boxes to OBJ files.

First-principles view:
- Input signal: pixels + camera geometry metadata.
- Representation change: image feature maps -> BEV feature map.
- Decision layer: dense prediction head + NMS.
- Output contract: structured 3D boxes and scores.
- Rendering contract: geometry serialized into files.

## 2) System map (high level)

Use this mental model:
1. Entry script parses args and starts runtime.
2. API layer builds model and pipeline.
3. Pipeline loads/transforms data into model inputs.
4. Detector runs forward pass and returns boxes/scores/labels.
5. Visualization layer filters by score and writes geometry files.

For this repository, the path is:
1. `demo/pcd_demo.py`
2. `mmdet3d/apis/inference.py`
3. `mmdet3d/models/detectors/bevdet.py`
4. `mmdet3d/models/detectors/centerpoint.py`
5. `mmdet3d/core/visualizer/show_result.py`

## 3) Exact visualization call chain in this codebase

### 3.1 Script entry

`demo/pcd_demo.py`
- Parses `pcd`, `config`, `checkpoint`, `score-thr`, `out-dir`, `show`.
- Calls `init_model(...)`.
- Calls `inference_detector(model, args.pcd)`.
- Calls `show_result_meshlab(...)`.

### 3.2 Model init + inference orchestration

`mmdet3d/apis/inference.py`
- `init_model(...)`
  - Loads config and checkpoint.
  - Builds model from config registry.
  - Sets eval mode.
- `inference_detector(...)`
  - Builds test pipeline from `model.cfg.data.test.pipeline`.
  - Prepares a `data` dict (points path or points tensor).
  - Applies pipeline, collates batch, scatters to device.
  - Runs `model(return_loss=False, rescale=True, **data)`.
  - Returns `(result, data)`.

### 3.3 Visualization API

`mmdet3d/apis/inference.py`
- `show_result_meshlab(...)`
  - Validates task type.
  - For detection task (`task='det'`), calls `show_det_result_meshlab(...)`.
  - Inside that function, predicted boxes are filtered by `score_thr`.
  - Calls the low-level writer via `show_result(...)`.

### 3.4 File writer

`mmdet3d/core/visualizer/show_result.py`
- `show_result(...)`
  - Creates output directory: `<out_dir>/<filename>/`.
  - Writes points as OBJ with `_write_obj(...)`.
  - Converts box center convention (bottom-center to gravity-center).
  - Writes GT and predicted oriented boxes with `_write_oriented_bbox(...)`.
- Output files typically include:
  - `<filename>_points.obj`
  - `<filename>_pred.obj`
  - `<filename>_gt.obj` (if GT provided)

## 4) How BEVDet internals produce those boxes

`mmdet3d/models/detectors/bevdet.py`
- `extract_img_feat(...)`
  - `prepare_inputs(...)`: builds geometry transforms (sensor to key ego).
  - `image_encoder(...)`: camera images -> CNN features.
  - `img_view_transformer(...)`: projects image features to BEV space.
  - `bev_encoder(...)`: BEV feature refinement.
- `simple_test(...)`
  - Calls `simple_test_pts(...)` from CenterPoint path.
  - Packs results under `result_dict['pts_bbox']`.

`mmdet3d/models/detectors/centerpoint.py`
- Performs head decoding and post-processing (including NMS) to produce final boxes/scores/labels used by visualization.

## 5) Config as executable architecture

`configs/bevdet/bevdet-r50.py` defines the model graph:
- `img_backbone`: ResNet image feature extractor.
- `img_neck`: feature pyramid aggregation.
- `img_view_transformer`: image -> BEV projection.
- `img_bev_encoder_backbone` and `img_bev_encoder_neck`: BEV refinement.
- `pts_bbox_head`: CenterPoint-style detection head.

It also defines:
- `train_pipeline` and `test_pipeline`.
- test-time score threshold and NMS behavior in `test_cfg`.

Important systems insight:
- Most behavior changes are config-driven, not hardcoded.
- The runtime API and model code are generic; config specializes them.

## 6) Data contract across layers

Think in contracts between modules:
1. Pipeline -> model:
   - tensors for points/images and metadata in `img_metas`.
2. Model -> post-process:
   - dense head outputs (heatmap/regression fields).
3. Post-process -> visualization:
   - `boxes_3d`, `scores_3d`, `labels_3d`.
4. Visualization -> filesystem:
   - OBJ geometry artifacts.

If output looks wrong, check contract boundaries in this order.

## 7) Beginner debugging checklist (visualization-first)

1. Confirm `--config` matches checkpoint family.
2. Confirm `task='det'` path is used in `show_result_meshlab`.
3. Lower `--score-thr` to verify predictions are not hidden by filtering.
4. Inspect `result` keys (`pts_bbox`, `boxes_3d`, `scores_3d`, `labels_3d`).
5. Confirm output folder `<out_dir>/<filename>/` exists.
6. Open `*_pred.obj` and `*_points.obj` in MeshLab/Open3D viewer.

## 8) Hands-on exercises (for understanding, not memorization)

### Exercise A: Explain one artifact
- Pick one generated `*_pred.obj` file.
- Explain where it is created and which function writes it.
- Explain why z-center is adjusted before writing.

### Exercise B: Threshold reasoning
- Run once with `--score-thr 0.0` and once with `--score-thr 0.3`.
- Compare number of boxes and explain the change from first principles.

### Exercise C: Architecture linkage
- Trace one predicted box backward:
  - visualization writer -> `show_result_meshlab` -> model output -> `BEVDet.simple_test`.
- Write a 5-line summary of each stage's responsibility.

## 9) Recommended reading order (90-120 minutes)

1. `demo/pcd_demo.py` (entry flow)
2. `mmdet3d/apis/inference.py` (`init_model`, `inference_detector`, `show_result_meshlab`)
3. `mmdet3d/core/visualizer/show_result.py` (file writing mechanics)
4. `configs/bevdet/bevdet-r50.py` (architecture + test behavior)
5. `mmdet3d/models/detectors/bevdet.py` (feature path)
6. `mmdet3d/models/detectors/centerpoint.py` (box decode/post-process)

## 10) Quick glossary

- BEV: Top-down feature representation over ground plane.
- View transformer: module that projects image features into BEV.
- `img_metas`: metadata used for geometric transforms and coordinate alignment.
- NMS: removes redundant overlapping predictions.
- Score threshold: removes low-confidence predictions before visualization.

## 11) What to edit for common goals

- Change visible predictions: score threshold in demo args or visualization call.
- Change detection behavior: `test_cfg` in `configs/bevdet/bevdet-r50.py`.
- Change architecture: `model` section in config.
- Change output file behavior: `mmdet3d/core/visualizer/show_result.py`.

---

If you can explain the data contract at each boundary, you understand the system.
