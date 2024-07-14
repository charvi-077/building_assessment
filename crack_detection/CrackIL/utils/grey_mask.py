# import cv2

# # Load the input image
# image = cv2.imread('../eval_data/masks1/26.png')

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Threshold the grayscale image
# _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

# # Invert the thresholded image
# # thresh = cv2.bitwise_not(thresh)

# # Save the resulting image
# cv2.imwrite('result_image.png', thresh)

# print("Resulting image saved successfully.")

import cv2
import os

# Input and output folder paths
input_folder = '../eval_data/crack_dataset1/labels'
output_folder = '../eval_data/crack_dataset1/output'

# Ensure the output folder exists, create if it doesn't
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
input_files = os.listdir(input_folder)

# Process each image
for filename in input_files:
    # Load the input image
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Save the resulting image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, thresh)

print("All images processed and saved successfully.")