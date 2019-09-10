"""
Global Parameters for the VGGish Model
======================================
"""

# --------------------------------------------------------------------------------------------------
# Architectural
# --------------------------------------------------------------------------------------------------

NUM_FRAMES = 96
NUM_BANDS = 64
EMBEDDING_SIZE = 128

# --------------------------------------------------------------------------------------------------
# Hyperparameters for feature generation
# --------------------------------------------------------------------------------------------------

NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
EXAMPLE_WINDOW_SECONDS = 0.96
EXAMPLE_STEP_SECONDS = 0.96

# --------------------------------------------------------------------------------------------------
# Embedding postprocessing
# --------------------------------------------------------------------------------------------------

PCA_EIGEN_VECTORS_NAME = "pca_eigen_vectors"
PCA_MEANS_NAME = "pca_means"
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

# --------------------------------------------------------------------------------------------------
# Hyperparameters used in training
# --------------------------------------------------------------------------------------------------

INIT_STDDEV = 0.01
LEARNING_RATE = 1e-4
ADAM_EPSILON = 1e-8

# --------------------------------------------------------------------------------------------------
# Names of ops, tensors and features
# --------------------------------------------------------------------------------------------------

INPUT_OP_NAME = "vggish/input_features"
INPUT_TENSOR_NAME = INPUT_OP_NAME + ":0"
OUTPUT_OP_NAME = "vggish/embedding"
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ":0"
AUDIO_EMBEDDING_FEATURE_NAME = "audio_embedding"
