import os
import sys
import argparse
from ultralytics import YOLO
import cv2
import time
import torch
import numpy as np

def download_models(model_names=["yolo11n.pt", "yolov8n.pt"]):
    """
    Download multiple YOLO models if they don't exist locally
    """
    models = {}
    for model_name in model_names:
        print(f"Loading {model_name} model...")
        try:
            model = YOLO(model_name)
            print(f"Model {model_name} loaded successfully")
            models[model_name] = model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            model_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}"
            print(f"Attempting to load from URL: {model_url}")
            try:
                model = YOLO(model_url)
                print(f"Model loaded successfully from URL")
                models[model_name] = model
            except Exception as e:
                print(f"Error loading model from URL: {e}")
                print(f"Skipping model {model_name}")
    
    return models

def compare_models(models, camera_id=0, conf_threshold=0.25, duration=30, img_size=640, save_output=False, output_dir=None):
    """
    Compare multiple YOLO models on webcam feed with optimized settings
    
    Args:
        models: Dictionary of YOLO model instances
        camera_id: Camera device ID (default is 0 for the primary webcam)
        conf_threshold: Confidence threshold for detections
        duration: Duration in seconds to test each model
        img_size: Input image size (must be multiple of 32)
        save_output: Whether to save the output videos
        output_dir: Directory to save the output videos
    """
    if not models:
        print("No models to compare")
        return
    
    # Ensure image size is multiple of 32
    img_size = (img_size // 32) * 32
    
    # Create output directory if saving
    if save_output:
        if output_dir is None:
            output_dir = "model_comparison"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Testing model: {model_name}")
        print(f"{'='*50}")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open webcam with ID {camera_id}")
            continue
        
        # Set camera resolution to match input size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size)
        
        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer if saving
        out = None
        if save_output:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f"{os.path.splitext(model_name)[0]}_{img_size}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        # Performance metrics
        frame_count = 0
        start_time = time.time()
        inference_times = []
        
        print(f"Testing {model_name} for {duration} seconds. Press 'q' to skip to next model.")
        print(f"Using image size: {width}x{height}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
            
            frame_count += 1
            
            # Perform detection and measure inference time
            inference_start = time.time()
            results_pred = model.predict(frame, conf=conf_threshold, verbose=False, augment=False, imgsz=img_size)[0]
            inference_end = time.time()
            inference_time = (inference_end - inference_start) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Visualize results (minimal overhead)
            annotated_frame = results_pred.plot()
            
            # Add minimal info overlay
            cv2.putText(annotated_frame, f"FPS: {1/(inference_time/1000):.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save frame if needed
            if out is not None:
                out.write(annotated_frame)
            
            # Display frame
            cv2.imshow("YOLO Model Comparison", annotated_frame)
            
            # Break if duration reached or 'q' pressed
            if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > duration:
                break
        
        # Release resources
        cap.release()
        if out is not None:
            out.release()
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_inference = np.mean(inference_times) if inference_times else 0
        
        # Store results
        results[model_name] = {
            "frames": frame_count,
            "time": elapsed_time,
            "avg_fps": avg_fps,
            "avg_inference_ms": avg_inference,
            "img_size": img_size
        }
        
        print(f"Test completed for {model_name}")
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average inference time: {avg_inference:.2f}ms")
    
    # Close windows
    cv2.destroyAllWindows()
    
    # Print comparison summary
    print("\n\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<15} | {'Img Size':<10} | {'Avg FPS':<10} | {'Avg Inference (ms)':<20} | {'Frames':<10}")
    print("-"*70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} | {metrics['img_size']:<10} | {metrics['avg_fps']:<10.2f} | {metrics['avg_inference_ms']:<20.2f} | {metrics['frames']:<10}")
    
    print("="*70)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare different YOLO models on webcam feed with optimized settings")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds for each model (default: 30)")
    parser.add_argument("--img", type=int, default=640, help="Input image size (must be multiple of 32, default: 640)")
    parser.add_argument("--save", action="store_true", help="Save the output videos")
    parser.add_argument("--output", type=str, help="Directory to save the output videos (optional)")
    parser.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolov8n.pt"], 
                        help="List of models to compare (default: yolo11n.pt yolov8n.pt)")
    
    args = parser.parse_args()
    
    # Download/load the models
    models = download_models(args.models)
    
    # Compare the models
    compare_models(models, args.camera, args.conf, args.duration, args.img, args.save, args.output)

if __name__ == "__main__":
    main() 