import numpy as np

def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    nearest_idx = []

    dot_prod = np.dot(matrix, vector)
    norm_rows = np.linalg.norm(matrix, axis=1)
    norm_vector = np.linalg.norm(vector)
    cosine_sim = dot_prod / (norm_rows * norm_vector)
    nearest_idx = np.argsort(-cosine_sim)[0:k]
    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    matrix = [[1,2,3,4],[0,0,1,0],[1,4,2,0]]
    vector = [1,2,3,4]
    nn = knn(vector, matrix, k=2)
    print nn
    # assert nn == [0,2]

if __name__ == "__main__":
    test_knn()


