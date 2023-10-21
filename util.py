import numpy as np
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def are_bboxes_similar(bbox1, bbox2, threshold):
    return all(euclidean_distance(p1, p2) <= threshold for p1, p2 in zip(bbox1, bbox2))
