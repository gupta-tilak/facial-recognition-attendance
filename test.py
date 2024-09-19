import cv2
from deepface import DeepFace
import numpy as np
from typing import Union, Tuple, Dict
from heapq import nlargest

# Define the align_img_wrt_eyes function
def align_img_wrt_eyes(
    img: np.ndarray,
    left_eye: Union[list, tuple],
    right_eye: Union[list, tuple],
) -> Tuple[np.ndarray, float]:
    """
    Align a given image horizontally with respect to their left and right eye locations
    Args:
        img (np.ndarray): pre-loaded image with detected face
        left_eye (list or tuple): coordinates of left eye with respect to the person itself
        right_eye (list or tuple): coordinates of right eye with respect to the person itself
    Returns:
        img (np.ndarray): aligned facial image
        angle (float): rotation angle used for alignment
    """
    if left_eye is None or right_eye is None:
        return img, 0

    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0

    angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return img, angle

# Define the expand_and_align_face function
def expand_and_align_face(
    facial_area: Dict[str, int], img: np.ndarray,
    align: bool, expand_percentage: int = 0, width_border: int = 0,
    height_border: int = 0) -> np.ndarray:
    
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
    left_eye = facial_area.get('left_eye')
    right_eye = facial_area.get('right_eye')
    print(left_eye)
    print(right_eye)

    if expand_percentage > 0:
        expanded_w = w + int(w * expand_percentage / 100)
        expanded_h = h + int(h * expand_percentage / 100)

        x = max(0, x - int((expanded_w - w) / 2))
        y = max(0, y - int((expanded_h - h) / 2))
        w = min(img.shape[1] - x, expanded_w)
        h = min(img.shape[0] - y, expanded_h)

    detected_face = img[int(y):int(y + h), int(x):int(x + w)]

    if align:
        aligned_img, angle = align_img_wrt_eyes(img=detected_face, left_eye=left_eye, right_eye=right_eye)
        return aligned_img

    return detected_face

# Main function to extract and show the face
def extract_and_show_face(img_path, detector_backend: str, align: bool = True):
    # Load the image using OpenCV
    img = cv2.imread(img_path)

    # Detect faces using DeepFace's extract_faces method
    detected_faces = DeepFace.extract_faces(
        detector_backend=detector_backend,
        img_path=img_path,
        align=True,  # Alignment will be handled manually later
        enforce_detection=False
    )

    print(len(detected_faces))

    # Check if any faces were detected
    if isinstance(detected_faces, list) and len(detected_faces) > 0:
        # Select the face with the highest confidence
        face_info = detected_faces[len(detected_faces)-1]
        facial_area = face_info['facial_area']
        confidence = face_info['confidence']

        print(facial_area)
        print(confidence)

        # Expand and align the detected face
        aligned_face = expand_and_align_face(
            facial_area=facial_area, 
            img=img, 
            align=align, 
            expand_percentage=0  # You can adjust this as needed
        )

        # Display the aligned face
        cv2.imshow("Aligned Face", aligned_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return facial_area, confidence  # Return the face info if needed
    else:
        print("No face detected.")
        return None

# Example usage
img_path = '/Users/guptatilak/Documents/C4GT-Face-Recognition/offline-FR/faceboxes-edgeface-FR/images/Aisvarrya_9.jpeg'
extract_and_show_face(img_path, detector_backend="mtcnn", align=True)
