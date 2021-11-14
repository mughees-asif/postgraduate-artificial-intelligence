from calculate_hypothesis import *


def hypothesis_to_vector(X, theta):
    hypothesis_vec = np.array([], dtype=np.float32)
    for i in range(X.shape[0]):
        hypothesis_vec = np.append(hypothesis_vec, calculate_hypothesis(X, theta, i))

    return hypothesis_vec
