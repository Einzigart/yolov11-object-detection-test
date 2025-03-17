import os
import sys
import argparse
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

def process_webcam(model, camera_id=0, conf_threshold=0.25, save_output=False, output_path=None):
    """
    Process webcam feed with YOLOv11s model
    
    Args:
        model: YOLO model instance
        camera_id: Camera device ID (default is 0 for the primary webcam)
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save the output video
        output_path: Path to save the output video (if None, will be auto-generated)
    """
    # Open the webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {camera_id}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Assuming 30 FPS for webcam
    
    # Create output video writer if save_output is True
    out = None
    if save_output:
        if output_path is None:
            output_path = f"webcam_detected_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    # Process the webcam feed
    frame_count = 0
    start_time = time.time()
    fps_display_time = start_time
    fps_counter = 0
    processing_fps = 0
    
    print("Processing webcam feed. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam")
            break
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate and display FPS every second
        current_time = time.time()
        if current_time - fps_display_time >= 1.0:
            processing_fps = fps_counter / (current_time - fps_display_time)
            fps_counter = 0
            fps_display_time = current_time
        
        # Perform detection
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        
        # Visualize the results on the frame
        annotated_frame = results.plot()
        
        # Add FPS information to the frame
        cv2.putText(annotated_frame, f"FPS: {processing_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write the frame to the output video if saving
        if out is not None:
            out.write(annotated_frame)
        
        # Display the frame
        cv2.imshow("YOLOv11 Webcam Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    print(f"Webcam processing ended. Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    if out is not None:
        print(f"Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11s detection on webcam feed")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save", action="store_true", help="Save the output video")
    parser.add_argument("--output", type=str, help="Path to save the output video (optional)")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Model name (default: yolo11s.pt)")
    
    args = parser.parse_args()
    
    # Download/load the model
    model = download_model(args.model)
    
    # Create output directory if saving and output path is specified
    if args.save and args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Process the webcam feed
    process_webcam(model, args.camera, args.conf, args.save, args.output)

if __name__ == "__main__":
    main() 