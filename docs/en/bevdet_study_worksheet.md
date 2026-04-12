# BEVDet Study Worksheet (Visualization-First)

Use this worksheet while reading code and running one demo inference.

## Session 1: Pipeline map

Goal: explain the end-to-end flow in your own words.

1. Entry script used:
   - File:
   - Main arguments:
2. Inference API function called:
   - Function name:
   - What it returns:
3. Visualization API function called:
   - Function name:
   - Which threshold is applied:
4. Final writer function:
   - Function name:
   - Output folder pattern:

Your one-sentence flow:


## Session 2: Data contracts

Goal: identify what each layer expects and produces.

1. Pipeline output keys before model call:
2. Model prediction keys used by visualization:
3. What each predicted box contains (fields):
4. Why coordinate conversion is needed before OBJ export:

Contract summary:


## Session 3: Controlled experiment

Goal: connect theory with behavior.

Run two cases:
- Case A: score-thr = 0.0
- Case B: score-thr = 0.3

Record:
1. Number of visible predicted boxes in A:
2. Number of visible predicted boxes in B:
3. Hypothesis before running:
4. Explanation after running:

What changed and why:


## Session 4: Architecture linkage

Goal: connect outputs to model design.

From config, list these modules:
1. img_backbone:
2. img_neck:
3. img_view_transformer:
4. img_bev_encoder_backbone:
5. img_bev_encoder_neck:
6. pts_bbox_head:

Now explain each module in one line based on where it appears in forward path.


## Session 5: System-level reflection

Goal: move from code reading to systems thinking.

1. Which subsystem controls behavior most strongly for your task?
2. Which boundary is most fragile (data/prediction/rendering) and why?
3. If output is empty, what is your top-3 debug order?
4. What one change would you make safely first?

Final reflection (5-8 lines):


## Completion criteria

You can say you understand the visualization pipeline when you can do all of these without looking up notes:
1. Describe call chain from script to OBJ files.
2. Explain where and why threshold filtering happens.
3. Explain why BEV representation is useful for driving.
4. Point to one file to change output behavior and one file to change model behavior.
