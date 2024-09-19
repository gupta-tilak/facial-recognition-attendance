import pandas as pd
import os
import cv2
import time
import numpy as np
import psutil
from compare_images import compare_images
from tqdm import tqdm

# Load the CSV file
csv_path = '/Users/guptatilak/Documents/C4GT-Face-Recognition/Dataset/An Indian facial database highlighting the Spectacle 2/Version 2/test_combinations_without_specs.csv'
data = pd.read_csv(csv_path)
n_pairs = 500

data = data.sample(n_pairs)

# Function to check illumination in the image
def check_illumination(filename):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print(f'ERROR: Unable to load {filename}')
    # Calculate mean brightness as a percentage
    meanpercent = np.mean(im) * 100 / 255

    return meanpercent > 15

# Function to denoise an image using Gaussian Blur
def denoise_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            denoised_img = cv2.GaussianBlur(img, (3, 3), 0.5)
            return denoised_img
        else:
            print(f"Error reading image: {image_path}")
            return None
    except (IOError, SyntaxError, ValueError) as e:
        print(f"Error processing image: {image_path} ({e})")
        return None

detector_backend = 'faceboxes'
model_name = 'edgeface'
distance_metric = 'cosine'
align = False

alignment_text = "aligned" if align else "unaligned"
task = f"{model_name}_{detector_backend}_{distance_metric}_{alignment_text}"
output_file = f"threshold/outputs/{task}.csv"

distances = []
truth_values = []
times = []
time_gaussian = []
cpu_memory_usage = []
ram_usage = []

for index, row in tqdm(data.head(n_pairs).iterrows(), total=n_pairs, desc=task):
    img1_target = row['img1_path']
    img2_target = row['img2_path']
    truth_value = row['truth_value']

    if check_illumination(img1_target) and check_illumination(img2_target):
        # Track RAM usage before processing
        process = psutil.Process(os.getpid())
        start_ram_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        start_gaussian_time = time.time()
        denoised_img1 = denoise_image(img1_target)
        denoised_img2 = denoise_image(img2_target)
        end_gaussian_time = time.time()

        if denoised_img1 is None or denoised_img2 is None:
            continue  # Skip this pair if any image couldn't be processed

        start_time = time.time()
        result = compare_images(img1_target, img2_target, distance_metric=distance_metric)
        end_time = time.time()

        end_ram_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        distance = result
        distances.append(distance)
        truth_values.append(truth_value)
        times.append(end_time - start_time)
        time_gaussian.append(end_gaussian_time - start_gaussian_time)

        cpu_memory_usage.append(psutil.cpu_percent())
        ram_usage.append(end_ram_usage - start_ram_usage)

# Calculate averages
avg_time_per_pair = sum(times) / len(times) if times else 0
avg_time_per_pair_gaussian = sum(time_gaussian) / len(time_gaussian) if time_gaussian else 0
avg_cpu_memory = sum(cpu_memory_usage) / len(cpu_memory_usage) if cpu_memory_usage else 0
avg_ram_usage = sum(ram_usage) / len(ram_usage) if ram_usage else 0

print(f"Average time per pair of images for {task}: {avg_time_per_pair:.4f} seconds")
print(f"Average time for Gaussian Blur: {avg_time_per_pair_gaussian:.4f} seconds")
print(f"Average CPU memory usage: {avg_cpu_memory:.2f}%")
print(f"Average RAM usage per image combination: {avg_ram_usage:.2f} MB")

# Save results
df = pd.DataFrame({
    "actuals": truth_values,
    "distances": distances,
    "time_taken": times,
    "cpu_memory_usage": cpu_memory_usage,
    "ram_usage": ram_usage
})
df.to_csv(output_file, index=False)
