# YOLOv11s Video and Image Detection

This project uses the YOLOv11s model from Ultralytics to perform object detection on video and image files.
The main purpose of this project is to test and learn to do object detection using YOLOv11s model. 

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Place your video and image files in the `samples` directory.

## Usage

Run the detection script:

```bash
python detect_video.py
```

The script will:
- Automatically download the YOLOv11s model if it doesn't exist
- Process all video and image files in the `samples` directory
- Save the detection results in the `output` directory

## Features

- Processes both videos and images
- Shows progress information during video processing
- Automatically creates output directory
- Configurable confidence threshold for detections

## Model Information

This project uses the YOLOv11s model from Ultralytics. YOLOv11 is a state-of-the-art object detection model that offers improved accuracy and speed compared to previous versions.

For more information about YOLOv11, visit: [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)

## Supported File Types

- Videos: .mp4, .avi, .mov
- Images: .jpg, .jpeg, .png

## Customization

You can modify the `detect_video.py` script to:
- Change the confidence threshold (default: 0.25)
- Use a different YOLO model
- Customize the output file naming
- Enable real-time display during processing (uncomment the cv2.imshow lines) 