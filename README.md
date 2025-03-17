# YOLOv11s Video, Image, and Webcam Detection

This project uses the YOLOv11s model from Ultralytics to perform object detection on video files, image files, and webcam feeds.
The main purpose of this project is to test and learn to do object detection using YOLOv11s model. 

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Place your video and image files in the `samples` directory.

## Usage

### Process Video and Image Files

Run the detection script:

```bash
python detect_video.py
```

The script will:
- Automatically download the YOLOv11s model if it doesn't exist
- Process all video and image files in the `samples` directory
- Save the detection results in the `output` directory

### Process a Single File

To process a specific file:

```bash
python run_detection.py --file path/to/your/file.mp4 --output path/to/output.mp4 --conf 0.25
```

### Webcam Detection

To run object detection on your webcam feed:

```bash
python webcam_detection.py
```

Optional arguments:
- `--camera`: Camera device ID (default: 0)
- `--conf`: Confidence threshold (default: 0.25)
- `--save`: Save the output video
- `--output`: Path to save the output video
- `--model`: Model name (default: yolo11s.pt)

Example:
```bash
python webcam_detection.py --camera 0 --conf 0.3 --save --output webcam_output.mp4
```

Press 'q' to quit the webcam detection.

## Features

- Processes videos, images, and webcam feeds
- Shows progress information during processing
- Automatically creates output directory
- Configurable confidence threshold for detections
- Real-time FPS display for webcam detection

## Model Information

This project uses the YOLOv11s model from Ultralytics. YOLOv11 is a state-of-the-art object detection model that offers improved accuracy and speed compared to previous versions.

For more information about YOLOv11, visit: [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/models/yolo11/)

## Supported File Types

- Videos: .mp4, .avi, .mov
- Images: .jpg, .jpeg, .png
- Webcam: Live feed from connected camera devices

## Customization

You can modify the scripts to:
- Change the confidence threshold (default: 0.25)
- Use a different YOLO model
- Customize the output file naming
- Enable/disable real-time display during processing 