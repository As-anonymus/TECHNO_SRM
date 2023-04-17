import torch
import cv2
from yolov5.models import YoloV5
from yolov5.utils import detect_utils

# Load the model
model = YoloV5(backbone="yolov5s", num_classes=1)
model.load_state_dict(torch.load("vehicle_damage_detection.pth"))
model.eval()

# Load the image
img = cv2.imread("image.jpg")

# Perform vehicle damage detection
result = detect_utils.detect_image(img, model, conf_thres=0.5, iou_thres=0.5)

# Display
cv2.imshow("Vehicle Damage Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()