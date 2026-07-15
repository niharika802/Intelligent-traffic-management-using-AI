import numpy as np

class TrafficAnalyzer:
    def __init__(self, max_capacity=40, stop_line_y=400):
        """Calculates flow density configurations and spatial line queue approximations."""
        self.max_capacity = max_capacity
        self.stop_line_y = stop_line_y  # Pixel row baseline simulating the intersection stop line

    def calculate_metrics(self, detected_vehicles):
        """Computes current capacity saturation and physical queue distance."""
        current_count = len(detected_vehicles)
        
        # Calculate core density calculation percentage
        density = min((current_count / self.max_capacity), 1.0)
        
        max_queue_pixels = 0
        
        for vehicle in detected_vehicles:
            bbox = vehicle['bbox']
            # Bottom coordinate of the vehicle bounding box
            bottom_y = bbox[3] 
            
            # If the vehicle is trailing behind the stop line, estimate its distance
            if bottom_y < self.stop_line_y:
                distance_from_line = self.stop_line_y - bottom_y
                if distance_from_line > max_queue_pixels:
                    max_queue_pixels = distance_from_line
                    
        # Rough scale mapping factor: Convert pixels to approximate meters
        queue_length_meters = round(max_queue_pixels * 0.25, 2)
        
        return current_count, density, queue_length_meters