from __future__ import print_function, absolute_import

from .classification import accuracy
from .reid import ReIDEvaluator, PrecisionRecall, np_cosine_dist, np_euclidean_dist
from .rank import fast_evaluate_rank, fast_evaluate_rank_prcc, fast_evaluate_rank_cc
from .metric import tensor_euclidean_dist, tensor_cosine_dist
from .distance import compute_distance_matrix