import cv2
import sys
from detection import VehicleDetector
from density import TrafficAnalyzer
from signal_control import AdaptiveSignalController

def run_pipeline(video_source=0):
    """Orchestrates video extraction, computer vision processing, and smart signal routing."""
    # Initialize components
    print("[SYSTEM] Booting Intelligent Traffic Framework modules...")
    detector = VehicleDetector()
    analyzer = TrafficAnalyzer(max_capacity=30, stop_line_y=450)
    controller = AdaptiveSignalController()
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream source: {video_source}")
        print("[SYSTEM] Switching to demo synthetic mock interface instead.")
        run_mock_demo(analyzer, controller)
        return

    print("[SYSTEM] Pipeline online. Processing frames. Press 'q' inside video screen to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video feed ended or stream disconnected.")
            break
            
        # 1. Processing and Detection Pipeline (Notion Steps 3, 4, 5 & 6)
        # Optional manual frame resizing can be handled here if needed: frame = cv2.resize(frame, (640, 480))
        vehicles, emergency_flag = detector.process_frame(frame)
        
        # 2. Extract Count, Density, and Queues (Notion Steps 7, 9 & 10)
        count, density, queue_m = analyzer.calculate_metrics(vehicles)
        
        # 3. Compute Adaptive Light Splits via Machine Learning (Notion Step 13)
        green_duration, _ = controller.get_optimal_timing(count, density, queue_m, emergency_flag)
        
        # 4. Draw Virtual Stop Line and Bounding Boxes
        cv2.line(frame, (0, analyzer.stop_line_y), (frame.shape[1], analyzer.stop_line_y), (0, 0, 255), 2)
        for v in vehicles:
            bx = v['bbox']
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{v['id']}", (bx[0], bx[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 5. Render HUD UI Dashboard Overlay (Notion Step 18)
        hud_color = (0, 0, 255) if emergency_flag else (255, 0, 0)
        cv2.rectangle(frame, (10, 10), (450, 150), (0, 0, 0), -1)
        cv2.putText(frame, "INTELLIGENT TRAFFIC MANAGEMENT HUD", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Vehicles Counted: {count} | Density: {density*100:.1f}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Estimated Queue Max: {queue_m} meters", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"AI ALLOCATED TIMING: {green_duration} SECONDS", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 2)

        cv2.imshow("Intelligent Traffic Management System - AI Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Video streaming context released cleanly.")

def run_mock_demo(analyzer, controller):
    """Simulates pipeline execution using manual console data if a physical video file is missing."""
    print("\n--- RUNNING TELEMETRY SIMULATION MODE ---")
    mock_scenarios = [
        {"count": 5, "emergency": False, "desc": "Off-peak empty road"},
        {"count": 28, "emergency": False, "desc": "Peak rush-hour congestion buildup"},
        {"count": 12, "emergency": True, "desc": "Ambulance approaching lane corridor"}
    ]
    
    for case in mock_scenarios:
        # Simulate tracking logs internally
        vehicles_mock = [{'id': idx, 'bbox': [100, 200, 150, 450 - (idx * 15)]} for idx in range(case['count'])]
        c, d, q = analyzer.calculate_metrics(vehicles_mock)
        t, act = controller.get_optimal_timing(c, d, q, case['emergency'])
        
        print(f"\nScenario Test Profile: {case['desc']}")
        print(f"-> Processing Engine Results: Count={c}, Density={d*100:.1f}%, Target Queue={q}m")
        print(f"-> Selected Traffic Phase Output Plan: Dynamic Green Time = {t} seconds")
        print(f"-> Controller Log message: {act}")

if __name__ == "__main__":
    # To run on a recorded file, replace 0 with the file path string: run_pipeline("videos/traffic.mp4")
    # Pass 0 to run via your hardware system's default integrated webcam
    source_feed = 0
    if len(sys.argv) > 1:
        source_feed = sys.argv[1]
    run_pipeline(source_feed)