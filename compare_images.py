import torch
import numpy as np
from embeddings import load_model, extract_embedding
from typing import Union

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
    source_representation = source_representation.flatten()
    test_representation = test_representation.flatten()

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
def compare_images(image1_path, image2_path, model_name="edgeface_s_gamma_05", face_model='weights/FaceBoxes.pth', distance_metric="euclidean"):
    """
    Compare two images by calculating the distance between their embeddings.
    
    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        model_name (str): Name of the model to use for extracting embeddings.
        face_model (str): Path to the face detection model weights.
        distance_metric (str): Metric to calculate distance ("cosine", "euclidean", "euclidean_l2").
    
    Returns:
        distance (np.float64 or None): Calculated distance between the embeddings of the two images,
                                       or None if embeddings couldn't be computed.
    """
    try:
        # Load the model
        model = load_model(model_name)
        
        # Extract embeddings using the function from the external module
        embedding1 = extract_embedding(model, image1_path, face_model=face_model)
        embedding2 = extract_embedding(model, image2_path, face_model=face_model)

        if embedding1 is not None and embedding2 is not None:
            # Convert embeddings to numpy arrays if they are PyTorch tensors
            if isinstance(embedding1, torch.Tensor):
                embedding1 = embedding1.detach().cpu().numpy()
            if isinstance(embedding2, torch.Tensor):
                embedding2 = embedding2.detach().cpu().numpy()

            # Compute the distance using the chosen metric
            distance = find_distance(embedding1, embedding2, distance_metric)
            return distance
        else:
            raise ValueError("Failed to compute embeddings for one or both images.")
    
    except ValueError as e:
        print(f"Error: {e}")
        return None

def save_embeddings_to_csv(embeddings, user_id, file_path):
    """
    Saves embeddings along with the user ID into a CSV file.
    
    Args:
        embeddings (np.ndarray): The embeddings to save.
        user_id (str): The identifier for the user.
        file_path (str): The path to the CSV file where embeddings will be saved.
    """
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_id] + embeddings.tolist())

def calculate_max_cosine_similarity(template_embeddings, test_embedding):
    """
    Calculates the maximum cosine similarity between the template embeddings and the test embedding.
    
    Args:
        template_embeddings (list of np.ndarray): List of template embeddings.
        test_embedding (np.ndarray): The test embedding.
    
    Returns:
        max_similarity (float): The maximum cosine similarity value.
    """
    max_similarity = -1
    for template_embedding in template_embeddings:
        similarity = 1 - find_cosine_distance(template_embedding, test_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
    return max_similarity
    