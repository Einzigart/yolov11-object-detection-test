import os
import argparse
from detect_video import download_model, process_video, process_image

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11s detection on a specific file")
    parser.add_argument("--file", type=str, required=True, help="Path to the video or image file")
    parser.add_argument("--output", type=str, help="Path to save the output file (optional)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Model name (default: yolo11s.pt)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return
    
    # Download/load the model
    model = download_model(args.model)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output) if args.output else "output"
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the file based on its extension
    file_ext = os.path.splitext(args.file)[1].lower()
    
    if file_ext in ['.mp4', '.avi', '.mov']:
        process_video(model, args.file, args.output, args.conf)
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        process_image(model, args.file, args.output, args.conf)
    else:
        print(f"Unsupported file type: {file_ext}")
        print("Supported types: .mp4, .avi, .mov, .jpg, .jpeg, .png")

if __name__ == "__main__":
    main() 