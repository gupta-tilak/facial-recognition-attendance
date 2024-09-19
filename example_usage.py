from compare_images import compare_images

# Define image paths
image1_path = 'images/Aisvarrya_8.jpeg'
image2_path = 'images/Rajath_14.jpg'

try:
    # Call the compare_images function from the imported module
    similarity = compare_images(image1_path, image2_path)
    print(f"Cosine Similarity: {similarity:.4f}")
except ValueError as e:
    print(e)
