import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def detect_and_match_features(img1, img2):
    # Use ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches

def estimate_transformation(kp1, kp2, matches):
    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H

def stitch_images(images):
    # Initialize the panorama
    panorama = images[0]
    for i in range(1, len(images)):
        img2 = images[i]
        
        # Detect and match features
        kp1, kp2, matches = detect_and_match_features(panorama, img2)
        
        # Estimate transformation
        H = estimate_transformation(kp1, kp2, matches)
        
        # Warp the second image
        height, width = img2.shape[:2]
        panorama_warped = cv2.warpPerspective(panorama, H, (width + panorama.shape[1], max(height, panorama.shape[0])))
        
        # Create a mask for blending
        mask = np.zeros((panorama_warped.shape[0], panorama_warped.shape[1]), dtype=np.uint8)
        mask[0:panorama.shape[0], 0:panorama.shape[1]] = 255
        
        # Blend the images
        panorama_warped = cv2.addWeighted(panorama_warped, 0.5, cv2.warpPerspective(img2, np.eye(3), (panorama_warped.shape[1], panorama_warped.shape[0])), 0.5, 0)
        
        # Update the panorama
        panorama = panorama_warped
    
    return panorama

def main():
    folder = '../utils/LEDNet/test/images/'  # Replace with your folder path
    images = load_images_from_folder(folder)
    
    if not images:
        print("No images found in the specified folder.")
        return
    
    panorama = stitch_images(images)
    
    # Save the stitched image
    cv2.imwrite('panorama.png', panorama)

if __name__ == '__main__':
    main()
