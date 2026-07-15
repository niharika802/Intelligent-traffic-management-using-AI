import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class AdaptiveSignalController:
    def __init__(self):
        """Initializes and trains the underlying Random Forest Decision model."""
        self.model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            min_samples_leaf=1,
            max_features='sqrt'
        )
        self._pre_train_model()

    def _pre_train_model(self):
        """Generates mock historical intersection logs to bootstrap the AI logic."""
        rng = np.random.default_rng(42)
        samples = 1000
        
        # Synthesize metrics matching Notion step 13
        vehicle_counts = rng.integers(0, 50, samples)
        densities = vehicle_counts / 40.0
        queue_lengths = vehicle_counts * 2.5 + rng.uniform(0, 10, samples)
        
        df = pd.DataFrame({
            'Count': vehicle_counts,
            'Density': densities,
            'Queue': queue_lengths
        })
        
        def assign_timing(row):
            if row['Density'] > 0.80 or row['Queue'] > 100:
                return 60
            elif row['Density'] > 0.50:
                return 45
            elif row['Density'] > 0.20:
                return 30
            else:
                return 20
                
        y = df.apply(assign_timing, axis=1)
        self.model.fit(df, y)

    def get_optimal_timing(self, vehicle_count, density, queue_length, emergency_flag):
        """Predicts dynamic signal durations and executes emergency preemptive overrides."""
        if emergency_flag:
            return 60, "EMERGENCY PREEMPTION ACTIVE: Priority corridor override triggered."
            
        # Format input feature vector for inference
        input_data = pd.DataFrame([[vehicle_count, density, queue_length]], 
                                  columns=['Count', 'Density', 'Queue'])
        
        # Predict optimal green phase allocation
        predicted_time = int(self.model.predict(input_data)[0])
        action = "Standard telemetry adaptive tracking timing applied."
        
        return predicted_time, action