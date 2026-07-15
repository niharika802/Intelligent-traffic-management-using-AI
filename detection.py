import cv2
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initializes the YOLOv8 detector using ultralytics pre-trained weights."""
        # This will automatically download yolov8n.pt if not locally present
        self.model = YOLO(model_path)
        
        # Class maps defined by default COCO dataset configurations
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # For prototype fallback, we tag buses/large vehicles or check labels for priority mapping
        self.emergency_keywords = ['ambulance', 'fire truck', 'police']

    def process_frame(self, frame):
        """Runs inference and returns tracking data along with emergency status flags."""
        # Using built-in ByteTrack/DeepSORT variant tracker inside Ultralytics
        results = self.model.track(frame, persist=True, verbose=False)
        detected_vehicles = []
        emergency_detected = False

        if not results or not results[0].boxes:
            return detected_vehicles, emergency_detected

        boxes = results[0].boxes
        for box in boxes:
            record, is_emergency = self._process_box(box)
            if record:
                detected_vehicles.append(record)
            if is_emergency:
                emergency_detected = True

        return detected_vehicles, emergency_detected

    def _process_box(self, box):
        """Process a single detection box. Returns (record_dict or None, emergency_flag)."""
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id not in self.vehicle_classes:
            return None, False

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        track_id = int(box.id[0]) if box.id is not None else 0

        record = {
            'id': track_id,
            'class_id': cls_id,
            'confidence': conf,
            'bbox': xyxy
        }

        is_emergency = (cls_id == 5 and conf > 0.85)
        return record, is_emergency