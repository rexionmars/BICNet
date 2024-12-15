import cv2
import os
import glob
import re

def natural_sort_key(s):
    """Key function for natural sorting of filename numbers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_rat_simulation_video(image_pattern='/home/rexionmars/estudos/MakeConciency/LOGS/complex_interaction_run_2/complex_interactions_*.png', output_name='complex_interaction.mp4', fps=15):
    """
    Creates a video from rat state images.
    
    Args:
        image_pattern (str): Pattern to match the image files
        output_name (str): Name of the output video file
        fps (int): Frames per second for the output video
    """
    # Get list of images and sort them naturally
    images = glob.glob(image_pattern)
    images.sort(key=natural_sort_key)
    
    if not images:
        print(f"No images found matching pattern: {image_pattern}")
        return
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    
    # Add frames to video
    for image in images:
        frame = cv2.imread(image)
        video.write(frame)
        print(f"Processing {image}")
    
    # Release video writer
    video.release()
    print(f"\nVideo created successfully: {output_name}")
    
if __name__ == "__main__":
    create_rat_simulation_video()