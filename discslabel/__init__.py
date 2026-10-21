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

# Anatomy segmentations (SEG_SPINE / SEG_CORD / SEG_CANAL) are combined into
# one int volume with non-overlapping label values. The ordering below defines
# the priority on voxel overlap: later keys win over earlier ones, so a voxel
# that belongs to both the spine and the canal is kept as CANAL.
SEG_LABEL_SPINE = 1
SEG_LABEL_CORD = 2
SEG_LABEL_CANAL = 3
# (key in dataset JSON, manifest / prepared sub-folder name, combined-volume value)
SEG_KEYS = (
    ("SEG_SPINE", "segs_spine", SEG_LABEL_SPINE),
    ("SEG_CORD", "segs_cord", SEG_LABEL_CORD),
    ("SEG_CANAL", "segs_canal", SEG_LABEL_CANAL),
)
