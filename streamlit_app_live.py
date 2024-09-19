import streamlit as st
import cv2
from PIL import Image
import os
import numpy as np
import pandas as pd
from embeddings import extract_embedding, load_model
from distance_metrics import compare_embeddings
import torch
import json
from datetime import datetime
import time

# Define the directories for saving images (registration and attendance)
REGISTER_DIR = 'attendance-data/register'
ATTENDANCE_DIR = 'attendance-data/mark_attendance'
TMP_DIR = os.path.join(ATTENDANCE_DIR, 'tmp')
REGISTER_CSV = 'attendance-data/registration_embeddings.csv'
ATTENDANCE_CSV = 'attendance-data/attendance_records.csv'

# Ensure the directories and CSV files exist
os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

def initialize_csv_files():
    if not os.path.exists(REGISTER_CSV):
        df_register = pd.DataFrame(columns=['user_id', 'image_number', 'embeddings'])
        df_register.to_csv(REGISTER_CSV, index=False)
    
    if not os.path.exists(ATTENDANCE_CSV):
        df_attendance = pd.DataFrame(columns=['user_id', 'attendance', 'timestamp'])
        df_attendance.to_csv(ATTENDANCE_CSV, index=False)

def save_images(user_id, images, save_dir):
    user_dir = os.path.join(save_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    image_paths = []
    for idx, image in enumerate(images):
        image_path = os.path.join(user_dir, f'image_{idx + 1}.jpg')
        image.save(image_path)
        image_paths.append(image_path)
    
    return image_paths, f"Images saved successfully for user ID: {user_id}"

def save_embeddings_to_csv(user_id, image_paths, model, face_model, csv_path):
    embeddings = []

    df = pd.read_csv(csv_path)
    
    for idx, image_path in enumerate(image_paths):
        embedding = extract_embedding(model, image_path, face_model)
        if embedding is not None:
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            embedding_list = embedding.tolist()  # Convert numpy array to list
            df = df.append({'user_id': user_id, 'image_number': idx + 1, 'embeddings': json.dumps(embedding_list)}, ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    
    return f"Embeddings saved successfully for user ID: {user_id}"

def register_user(user_id, images, model, face_model):
    # Save the uploaded images to the register directory and return their paths
    image_paths, response = save_images(user_id, images, REGISTER_DIR)
    
    # Save embeddings to the registration CSV
    save_embeddings_to_csv(user_id, image_paths, model, face_model, REGISTER_CSV)
    
    return response

from PIL import Image

# Function to mark attendance via webcam
def mark_attendance_live(user_id, model, face_model):
    df_register = pd.read_csv(REGISTER_CSV)
    
    if user_id in df_register['user_id'].values:
        registration_embeddings = df_register[df_register['user_id'] == user_id]['embeddings'].apply(json.loads).to_list()

        # Access webcam to capture the image
        cap = cv2.VideoCapture(0)
        st.text("Accessing webcam...")

        if not cap.isOpened():
            st.error("Webcam could not be opened.")
            return

        # Delay for 1 second to ensure the camera starts properly
        time.sleep(1)

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    st.error("Failed to capture image from webcam.")
                    break

                # Convert the frame to RGB (since OpenCV captures in BGR format)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save the image temporarily using PIL to ensure it's saved in RGB format
                temp_image_path = os.path.join(TMP_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                pil_image = Image.fromarray(image_rgb)
                pil_image.save(temp_image_path)

                # Extract the embedding for the captured image using the RGB image
                embedding = extract_embedding(model, temp_image_path, face_model)
                if embedding is not None:
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.detach().cpu().numpy()

                # Ensure embedding is a 1D vector
                embedding = embedding.flatten()

                # Compare the captured image embedding with all registered embeddings
                similarity_scores = []
                for reg_embedding in registration_embeddings:
                    reg_embedding_np = np.array(reg_embedding)
                    reg_embedding_np = reg_embedding_np.flatten()
                    similarity_score = compare_embeddings(reg_embedding_np, embedding, distance_metric='cosine')
                    similarity_scores.append(similarity_score)
                
                threshold = 0.25  # Set your threshold here
                attendance_marked = np.median([score >= threshold for score in similarity_scores]) >= 0.5
                
                if attendance_marked:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    user_dir = os.path.join(ATTENDANCE_DIR, str(user_id))
                    os.makedirs(user_dir, exist_ok=True)
                    image_path = os.path.join(user_dir, f'{timestamp}.jpg')
                    pil_image.save(image_path)  # Save the RGB image using PIL
                    
                    df_attendance = pd.read_csv(ATTENDANCE_CSV) if os.path.exists(ATTENDANCE_CSV) else pd.DataFrame(columns=['user_id', 'attendance', 'timestamp'])
                    df_attendance = df_attendance.append({
                        'user_id': user_id, 
                        'attendance': 1, 
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }, ignore_index=True)
                    
                    df_attendance.to_csv(ATTENDANCE_CSV, index=False)
                    
                    st.success(f"Attendance marked successfully for user ID: {user_id}. Similarity scores: {similarity_scores}. Attendance: Marked")
                    break
                else:
                    #Generate a timestamp to use as the image filename
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                    # Save the uploaded image to the attendance directory with the timestamp as the filename
                    user_dir = os.path.join(ATTENDANCE_DIR, str(user_id)+'/false-recognition')
                    os.makedirs(user_dir, exist_ok=True)
                    image_path = os.path.join(user_dir, f'{timestamp}.jpg')
                    pil_image.save(image_path)  # Save the RGB image using PIL

                    df_attendance = pd.read_csv(ATTENDANCE_CSV) if os.path.exists(ATTENDANCE_CSV) else pd.DataFrame(columns=['user_id', 'attendance', 'timestamp'])
                    df_attendance = df_attendance.append({
                        'user_id': user_id, 
                        'attendance': 0, 
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }, ignore_index=True)
                    
                    df_attendance.to_csv(ATTENDANCE_CSV, index=False)
                    st.error(f"Attendance not marked for user ID: {user_id}. Similarity scores: {similarity_scores}. Attendance: Not Marked")

                # Sleep for 0.5 seconds before capturing the next frame
                time.sleep(0.5)

        except Exception as e:
            st.error(f"Failed to mark attendance: {e}")
        
        finally:
            cap.release()  # Release the webcam

    else:
        st.error(f"Failed to find registration embeddings for user ID: {user_id}")



def main():
    st.title("Face Recognition Attendance System")

    menu = ["Register User", "Mark Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)

    model_name = "edgeface_s_gamma_05"
    face_model = 'weights/FaceBoxes.pth'
    device = 'cpu'
    
    model = load_model(model_name, checkpoint_dir="checkpoints", device=device)
    initialize_csv_files()

    if choice == "Register User":
        st.subheader("Register a New User")
        user_id = st.text_input("Enter User ID")

        if user_id:
            df_register = pd.read_csv(REGISTER_CSV)
            if user_id in df_register['user_id'].values:
                st.warning(f"User ID '{user_id}' already exists in the registration database.")
                existing_embeddings = df_register[df_register['user_id'] == user_id]['embeddings'].values
                st.text(f"Existing Embeddings for '{user_id}': {existing_embeddings}")

        uploaded_files = st.file_uploader("Upload 3 Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        if len(uploaded_files) == 3:
            if st.button("Register"):
                images = [Image.open(file) for file in uploaded_files]
                
                try:
                    response = register_user(user_id, images, model, face_model)
                    st.success(response)
                except Exception as e:
                    st.error(f"Failed to register user: {e}")

    elif choice == "Mark Attendance":
        st.subheader("Mark Attendance")
        user_id = st.number_input("Enter User ID", min_value=0, step=1, format="%d")

        if st.button("Start Webcam and Mark Attendance"):
            try:
                mark_attendance_live(user_id, model, face_model)
            except Exception as e:
                st.error(f"Failed to mark attendance: {e}")

if __name__ == '__main__':
    main()
