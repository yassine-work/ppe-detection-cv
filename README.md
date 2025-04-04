# PPE Detection
A computer vision project to detect personal protective equipment (PPE) on people in a video feed using a fine-tuned YOLOv8 model.

## Description
This project uses a fine-tuned YOLOv8 model (`ppe.pt`) to detect PPE items (e.g., hardhats, masks, safety vests) on people in a video feed. It identifies both the presence and absence of PPE, using color-coded bounding boxes to indicate compliance (green for PPE present, red for PPE absent). I built this to learn object detection and model fine-tuning as part of my computer vision journey.

## How It Works
- **Detection**: A fine-tuned YOLOv8 model (`ppe.pt`) detects objects in each frame of the video, including people and PPE items.
- **Classification**: The model identifies specific classes, such as `Hardhat`, `Mask`, `Safety Vest`, `NO-Hardhat`, `NO-Mask`, and `NO-Safety Vest`.
- **Visualization**: The video feed shows:
  - Bounding boxes around detected objects with their class names and confidence scores.
  - Color-coded boxes:
    - Green (`0, 255, 0`) for `Hardhat`, `Mask`, and `Safety Vest` (indicating compliance).
    - Red (`255, 0, 0`) for `NO-Hardhat`, `NO-Mask`, and `NO-Safety Vest` (indicating non-compliance).
    - Blue (`0, 0, 255`) for other classes (e.g., `Person`, vehicles).

## Requirements
- Python 3.x
- Install dependencies:pip install -r requirements.txt
- Key libraries: `ultralytics`, `opencv-python`, `cvzone`.
- Fine-tuned YOLOv8 model: The script uses `ppe.pt`, which is not included due to its size. You can download it from [your preferred link, e.g., Google Drive] and place it in the `models` folder.
- Video file: Download the sample video [here](https://drive.google.com/file/d/1tBGERkWg3nmn-fMuZ6wX9KPrAr3ej_v5/view?usp=sharing) and place it in the `videos` folder as `ppe-2.mp4`.

## How to Run
1. Clone this repo
2. Navigate to the folder
3. Install the required dependencies:pip install -r requirements.txt
4. Download the sample video and fine-tuned model (see Requirements) and place them in the correct folders.
5. Run the script:
python PPEDetection.py
- The script uses `videos/ppe-2.mp4` by default. To use a different video, update the video path in `PPEDetection.py` (line: `cap = cv2.VideoCapture("../videos/ppe-2.mp4")`).

## Example Output
Watch a demo video of the PPE detection in action [here](https://drive.google.com/file/d/1RVCtLI1uBW-Gpl7iD8ym_hgMau4G7qzG/view?usp=sharing).

You can download the sample video [here](https://drive.google.com/file/d/1tBGERkWg3nmn-fMuZ6wX9KPrAr3ej_v5/view?usp=sharing) to see the exact setup.

## Notes
- This is a beginner project—expect some rough edges!
- The model (`ppe.pt`) was fine-tuned specifically for PPE detection, improving accuracy for the classes listed in the script.
- - The model (`ppe.pt`) was fine-tuned on a custom dataset to detect PPE-related classes, enhancing accuracy for this specific use case. However, the model file is not provided in this repository due to its size.
- The script can be modified to use a webcam by uncommenting the webcam code (`cap = cv2.VideoCapture(1)`) and commenting out the video file line.
- This one does not use a mask, as the fine-tuned model is designed to work across the entire frame.