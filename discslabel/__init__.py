"""discslabel: UNet heatmap regression for intervertebral disc landmark detection."""

# Disc label values used in the ground-truth 3D label volumes.
# Values 1-25 correspond to intervertebral discs; the fixed anchors are:
DISC_LABEL_C2C3 = 3
DISC_LABEL_C7T1 = 8
DISC_LABEL_T12L1 = 20
DISC_LABEL_L5S1 = 25

# Output heatmap channels of the model:
#   0: all visible discs (union of any label 1..25)
#   1: C2C3
#   2: C7T1
#   3: T12L1
#   4: L5S1
HEATMAP_CHANNELS = ("all_discs", "C2C3", "C7T1", "T12L1", "L5S1")
NUM_CLASSES = len(HEATMAP_CHANNELS)
LANDMARK_CHANNEL_LABELS = (None, DISC_LABEL_C2C3, DISC_LABEL_C7T1, DISC_LABEL_T12L1, DISC_LABEL_L5S1)

# Anatomy segmentations are packed into a 4-channel BINARY tensor and fed as
# the second parameter of SmaugLab's AugTransformsGPU. One channel per
# anatomical structure, with no value overlap between channels — all channels
# are {0, 1} masks, so kornia's "mask" data-key handles them with nearest
# resampling under spatial transforms.
#
# SEG_SPINE on disk is a dense discrete segmentation that mixes vertebrae
# (low label values) with discs (high label values); the dataset reader
# splits those two populations by a configurable threshold
# (``SEG_SPINE_DISC_MIN_LABEL`` below).
SEG_CHANNELS = ("vertebrae", "discs", "spinal_cord", "spinal_canal")
NUM_SEG_CHANNELS = len(SEG_CHANNELS)
SEG_CH_VERTEBRAE = 0
SEG_CH_DISCS = 1
SEG_CH_CORD = 2
SEG_CH_CANAL = 3

# Default threshold used by the dataset reader to split SEG_SPINE voxels into
# vertebrae vs. discs: values < threshold are vertebrae, >= threshold are
# discs. The whole-spine convention uses 11..50 for vertebrae and 60+ for
# discs, so 60 cleanly separates them. Override per-run via the pipeline
# config field ``spine_disc_min_label``.
SEG_SPINE_DISC_MIN_LABEL = 60

# (key in dataset JSON, manifest key, prepared sub-folder name)
SEG_KEYS = (
    ("SEG_SPINE", "seg_spine", "segs_spine"),
    ("SEG_CORD", "seg_cord", "segs_cord"),
    ("SEG_CANAL", "seg_canal", "segs_canal"),
)
