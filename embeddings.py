import torch
from torchvision import transforms
from backbones import get_model
from detection import detect_faces

# Function to load the model
def load_model(model_name="edgeface_s_gamma_05", checkpoint_dir="checkpoints", device='cpu'):
    model = get_model(model_name)
    checkpoint_path = f'{checkpoint_dir}/{model_name}.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

# Function to extract embeddings
# Faceboxes as the detector model and EdgeFace as the recogntion model
def extract_embedding(model, image_path, face_model, device='cpu'):
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Detect face 
    detected_face = detect_faces(image_path, trained_model='weights/FaceBoxes.pth', cpu=True, 
                           confidence_threshold=0.05, top_k=5000, nms_threshold=0.3,
                           keep_top_k=750, vis_thres=0.5)
    
    if detected_face is None:
        print(f"Face detection failed for image: {image_path}")
        return None
    
    # Preprocess and add batch dimension
    transformed_input = transform(detected_face).unsqueeze(0).to(device)
    
    # Extract embedding
    with torch.no_grad():
        embedding = model(transformed_input)
    
    return embedding


# # Function to extract embeddings
# #Dlib as the recogntion model and faceboxes as the detector model
# import os
# import cv2
# from deepface import DeepFace

# import os
# import cv2
# from deepface import DeepFace

# def extract_embedding(model, image_path, face_model, device='cpu'):
#     # Define the directory to save the detected face
#     tmp_dir = 'threshold/tmp'
#     os.makedirs(tmp_dir, exist_ok=True)

#     # Detect face and align
#     detected_face = detect_faces(image_path, trained_model='weights/FaceBoxes.pth', cpu=True, 
#                            confidence_threshold=0.05, top_k=5000, nms_threshold=0.3,
#                            keep_top_k=750, vis_thres=0.5)
    
#     if detected_face is None:
#         print(f"Face detection failed for image: {image_path}")
#         return None
    
#     # Save the detected face
#     face_path = os.path.join(tmp_dir, os.path.basename(image_path))
#     cv2.imwrite(face_path, detected_face)

#     try:
#         # Use the saved face image for DeepFace.represent
#         embedding_objs = DeepFace.represent(
#             img_path=face_path,
#             model_name='Dlib',
#             enforce_detection=True  # Default is True; keeping it explicit here
#         )
#         return embedding_objs[0]['embedding']
    
#     except ValueError as e:
#         print(f"Skipping image {image_path} due to detection error: {e}")
#         return None

# # Function to extract embeddings
# # using yolov8 as the detector model and edgeface as recognition model
# import os
# import cv2
# import torch
# from torchvision import transforms
# from deepface import DeepFace
# from datetime import datetime
# import uuid
# import numpy as np

# def extract_embedding(model, image_path, face_model, device='cpu'):
#     # Define the transform
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
    
#     # Create a tmp directory if it doesn't exist
#     tmp_dir = 'tmp'
#     if not os.path.exists(tmp_dir):
#         os.makedirs(tmp_dir)
    
#     # Detect face using DeepFace
#     face_objs = DeepFace.extract_faces(img_path=image_path, detector_backend='yolov8', align=True)
    
#     if not face_objs:
#         print(f"Face detection failed for image: {image_path}")
#         return None
    
#     # Extract the first detected face
#     detected_face = face_objs[0]["face"]
    
#     # Scale the detected face to 0-255 range if needed
#     if detected_face.max() > 1.0:
#         detected_face = detected_face / detected_face.max()  # Normalize to 0-1 range
    
#     detected_face = (detected_face * 255).astype('uint8')  # Scale to 0-255 and convert to uint8
    
#     # # Check pixel values before saving
#     # print("Pixel values before saving:", np.min(detected_face), np.max(detected_face))
    
#     # Generate a unique filename using a timestamp and UUID
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     unique_id = str(uuid.uuid4())
#     tmp_image_path = os.path.join(tmp_dir, f'detected_face_{timestamp}_{unique_id}.jpg')
    
#     # Save the detected face to the tmp folder
#     cv2.imwrite(tmp_image_path, cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR))
    
#     # Load the image from the tmp folder
#     face_image = cv2.imread(tmp_image_path)
#     face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
#     # Preprocess and add batch dimension
#     transformed_input = transform(face_image).unsqueeze(0).to(device)
    
#     # Extract embedding
#     with torch.no_grad():
#         embedding = model(transformed_input)
    
#     return embedding



                   