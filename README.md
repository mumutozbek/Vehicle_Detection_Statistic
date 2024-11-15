# Vehicle Tracking and Speed Detection System

https://github.com/user-attachments/assets/637bfcc8-3681-4b5f-a4b0-da68b0e15a24


A real-time computer vision system for vehicle detection, tracking, and speed monitoring using YOLO and advanced computer vision techniques.

## Features

### Real-time Detection & Tracking
- Vehicle detection using YOLOv8
- Real-time speed estimation
- High-speed vehicle alerts (>40 km/h)
- Accurate IN/OUT counting
- Total vehicle tracking

### Advanced Analytics
- Speed monitoring and statistics
- Vehicle count tracking
- Traffic density analysis
- Peak hour detection
- Real-time statistics updates

### Modern Interface
- Dark-themed professional GUI
- Real-time video preview
- Live statistics dashboard
- Progress tracking
- Advanced analytics panel
- Automatic video saving

### Visual Alerts
- Blinking high-speed warnings
- Color-coded speed indicators
- Real-time count display
- Clear statistical visualization

  

https://github.com/user-attachments/assets/d8605c8b-817a-43e8-9963-26442b2d6efe




## Setup

1. Clone the repository:
```bash
https://github.com/mumutozbek/Vehicle_Detection_Statistic.git
cd Vehicle_Detection_Statistic
```

2. Create and activate virtual environment:
```bash
# For macOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Use the interface:
- Click "Select Video" to choose input video
- Click "Start Processing" to begin analysis
- Monitor real-time statistics and speed detections
- Processed video will be saved automatically

## Project Structure
```
vehicle-tracking-system/
├── src/
│   ├── interface/        # GUI components
│   ├── config/          # Configuration files
│   ├── utils/           # Utility functions
│   └── detectors/       # Detection models
├── data/
│   ├── input/           # Input videos
│   └── output/          # Processed videos
├── models/              # YOLO models
├── app_parking_management.py
├── main.py
└── requirements.txt
```

## Key Features Details

### Speed Detection
- Real-time speed estimation
- High-speed vehicle alerts (>40 km/h)
- Speed statistics tracking
- Visual speed indicators

### Vehicle Tracking
- Accurate vehicle detection
- IN/OUT counting
- Path tracking
- Total vehicle counting

### Analytics Dashboard
- Average speed display
- Maximum speed tracking
- Vehicle count statistics
- Traffic density analysis
- Peak hour detection

### Video Processing
- Automatic video saving
- Progress tracking
- Error handling
- Resource management

## Requirements
- Python 3.8+
- PyQt6
- OpenCV
- Ultralytics YOLO
- NumPy

## Output
- Processed videos saved with timestamp
- Format: `original_name_processed_YYYYMMDD_HHMMSS.mp4`
- Location: `data/output/`

## Author
[Mustafa Umut Ozbek](https://github.com/mumutozbek)

## License
MIT License

## Acknowledgments
- YOLOv8 (or you can try it with yolo11) for object detection
- PyQt6 for the user interface
- OpenCV for video processing

