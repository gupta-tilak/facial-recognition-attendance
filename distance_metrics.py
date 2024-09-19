from typing import Union
import torch
import numpy as np

# Distance Measuring Functions
def find_cosine_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns:
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    # Flatten the arrays to ensure they are 1D
    source_representation = source_representation
    test_representation = test_representation

    a = np.dot(source_representation, test_representation)
    b = np.linalg.norm(source_representation)
    c = np.linalg.norm(test_representation)
    return 1 - a / (b * c)


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    """
    Find euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    return np.linalg.norm(source_representation - test_representation)


def l2_normalize(x: Union[np.ndarray, list]) -> np.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
    Returns:
        y (np.ndarray): l2 normalized vector
    """
    if isinstance(x, list):
        x = np.array(x)
    norm = np.linalg.norm(x)
    return x if norm == 0 else x / norm


def find_distance(
    alpha_embedding: Union[np.ndarray, list],
    beta_embedding: Union[np.ndarray, list],
    distance_metric: str,
    ) -> np.float64:
    """
    Wrapper to find distance between vectors according to the given distance metric
    Args:
        alpha_embedding (np.ndarray or list): 1st vector
        beta_embedding (np.ndarray or list): 2nd vector
        distance_metric (str): distance metric ("cosine", "euclidean", "euclidean_l2")
    Returns:
        distance (np.float64): calculated distance
    """
    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        distance = find_euclidean_distance(
            l2_normalize(alpha_embedding), l2_normalize(beta_embedding)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return distance

# Compare Images Function
def compare_embeddings(embedding1, embedding2, distance_metric="cosine"):
    """
    Compare two images by calculating the distance between their embeddings.
    
    Args:
        embedding1 (str): Embedding of the first image.
        embedding2 (str): Embedding of second image.
        distance_metric (str): Metric to calculate distance ("cosine", "euclidean", "euclidean_l2").
    
    Returns:
        distance (np.float64 or None): Calculated distance between the embeddings of the two images,
                                       or None if embeddings couldn't be computed.
    """
    try:

        if embedding1 is not None and embedding2 is not None:
            # # Convert embeddings to numpy arrays if they are PyTorch tensors
            # if isinstance(embedding1, torch.Tensor):
            #     embedding1 = embedding1.detach().cpu().numpy()
            # if isinstance(embedding2, torch.Tensor):
            #     embedding2 = embedding2.detach().cpu().numpy()
            
            # Compute the distance using the chosen metric
            distance = find_distance(embedding1, embedding2, distance_metric)
            return distance
        else:
            raise ValueError("Failed to compute distances between embeddings for one or both images.")
    
    except ValueError as e:
        print(f"Error: {e}")
        return None