import os
import sys
from ultralytics import YOLO
import cv2
import time
import torch

def download_model(model_name="yolo11s.pt"):
    """
    Download the YOLOv11s model if it doesn't exist locally
    """
    print(f"Loading {model_name} model...")
    try:
        # Create a YOLO model instance which will download the model if needed
        model = YOLO(model_name)
        print(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try with the full URL path
        model_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}"
        print(f"Attempting to load from URL: {model_url}")
        try:
            model = YOLO(model_url)
            print(f"Model loaded successfully from URL")
            return model
        except Exception as e:
            print(f"Error loading model from URL: {e}")
            raise

def process_video(model, video_path, output_path=None, conf_threshold=0.25):
    """
    Process a video file with YOLOv11s model
    
    Args:
        model: YOLO model instance
        video_path: Path to the input video
        output_path: Path to save the output video (if None, will be auto-generated)
        conf_threshold: Confidence threshold for detections
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer if output_path is provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_detected.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process the video
    frame_count = 0
    start_time = time.time()
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Total frames: {total_frames}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), Processing FPS: {fps_processing:.1f}")
        
        # Perform detection
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        
        # Visualize the results on the frame
        annotated_frame = results.plot()
        
        # Write the frame to the output video
        out.write(annotated_frame)
        
        # Display the frame (optional - comment out for faster processing)
        # cv2.imshow("YOLOv11 Detection", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")

def process_image(model, image_path, output_path=None, conf_threshold=0.25):
    """
    Process a single image with YOLOv11s model
    
    Args:
        model: YOLO model instance
        image_path: Path to the input image
        output_path: Path to save the output image (if None, will be auto-generated)
        conf_threshold: Confidence threshold for detections
    """
    # Create output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_detected.jpg"
    
    # Perform detection
    results = model.predict(image_path, conf=conf_threshold)[0]
    
    # Save the annotated image
    annotated_img = results.plot()
    cv2.imwrite(output_path, annotated_img)
    
    print(f"Image processing complete. Output saved to {output_path}")

def main():
    print("Starting YOLOv11s detection script...")
    
    # Download/load the model
    model = download_model("yolo11s.pt")
    
    # Check if samples directory exists
    if not os.path.exists("samples"):
        print("Error: 'samples' directory not found")
        return
    
    print(f"Found samples directory: {os.path.abspath('samples')}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")
        print("Created output directory")
    
    # Process video samples
    video_samples = [f for f in os.listdir("samples") if f.endswith((".mp4", ".avi", ".mov"))]
    print(f"Found {len(video_samples)} video samples: {video_samples}")
    
    if video_samples:
        for video in video_samples:
            video_path = os.path.join("samples", video)
            output_path = os.path.join("output", f"{os.path.splitext(video)[0]}_detected.mp4")
            process_video(model, video_path, output_path)
    else:
        print("No video samples found in the 'samples' directory")
    
    # Process image samples
    image_samples = [f for f in os.listdir("samples") if f.endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(image_samples)} image samples: {image_samples}")
    
    if image_samples:
        for image in image_samples:
            image_path = os.path.join("samples", image)
            output_path = os.path.join("output", f"{os.path.splitext(image)[0]}_detected.jpg")
            process_image(model, image_path, output_path)
    else:
        print("No image samples found in the 'samples' directory")
    
    print("Detection script completed")

if __name__ == "__main__":
    main() 