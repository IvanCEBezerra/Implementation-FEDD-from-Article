import numpy as np

def cosine_distance(vec_a, vec_b):
    """
    Computes the cosine distance between two feature vectors.
    dist_cos(A, B) = 1 - (<A, B> / (||A|| * ||B||))
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def pearson_distance(vec_a, vec_b):
    """
    Computes the Pearson correlation distance between two feature vectors.
    dist_pear(A, B) = 1 - Corr(A, B)
    """
    if len(vec_a) < 2 or len(vec_b) < 2:
        return 0.0
        
    correlation = np.corrcoef(vec_a, vec_b)[0, 1]
    
    if np.isnan(correlation):
        return 0.0
        
    return correlation