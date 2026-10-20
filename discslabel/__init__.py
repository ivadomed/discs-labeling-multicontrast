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
