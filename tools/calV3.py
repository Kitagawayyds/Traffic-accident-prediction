import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Use default backend
matplotlib.use('TkAgg')  # Alternative: 'Qt5Agg'


# Risk calculation functions
def speed_risk(speed, speed_threshold=100):
    return min((speed / (speed_threshold * 1.3)) ** 4, 1) * 10


def fluctuation_risk(speed, fluctuation, fluctuation_ratio=0.5):
    dynamic_fluctuation_threshold = max(fluctuation_ratio * speed, 20)
    return min((fluctuation / dynamic_fluctuation_threshold) ** 2, 1) * 10


def angle_risk(angle, angle_threshold=90):
    return min((angle / angle_threshold) ** 2, 1) * 10


def overlap_risk(overlap, overlap_threshold=0.2):
    return min((overlap / overlap_threshold) ** 3, 1) * 10


# Create a range of values for speed from 0 to 200
speed_values = np.linspace(0, 200, 500)
speed_risk_scores = [speed_risk(s) for s in speed_values]

# Create a range of fluctuation values from 0 to 50 with a speed of 50
fluctuation_values = np.linspace(0, 50, 500)
fluctuation_risk_scores = [fluctuation_risk(50, f) for f in fluctuation_values]

# Create a range of angle values from 0 to 180 degrees
angles = np.linspace(0, 180, 500)
angle_risk_scores = [angle_risk(angle) for angle in angles]

# Create a range of overlap values from 0 to 1
overlap_values = np.linspace(0, 1, 500)
overlap_risk_scores = [overlap_risk(overlap) for overlap in overlap_values]

# Plotting the risk scores
plt.figure(figsize=(20, 6))  # Set the figure size

# Plot speed risk score subplot
plt.subplot(1, 4, 1)
plt.plot(speed_values, speed_risk_scores, label='Speed Risk Score', color='blue')
plt.xlabel('Speed')
plt.ylabel('Risk Score')
plt.title('Speed to Risk Score Mapping')
plt.grid(True)
plt.legend()

# Plot fluctuation risk score subplot
plt.subplot(1, 4, 2)
plt.plot(fluctuation_values, fluctuation_risk_scores, label='Fluctuation Risk Score', color='green')
plt.xlabel('Fluctuation')
plt.ylabel('Risk Score')
plt.title('Fluctuation Risk Score')
plt.grid(True)
plt.legend()

# Plot angle risk score subplot
plt.subplot(1, 4, 3)
plt.plot(angles, angle_risk_scores, label='Angle Risk Score', color='red')
plt.xlabel('Angle Change (degrees)')
plt.ylabel('Risk Score')
plt.title('Angle Risk Score vs Angle Change')
plt.grid(True)
plt.legend()

# Plot overlap risk score subplot
plt.subplot(1, 4, 4)
plt.plot(overlap_values, overlap_risk_scores, label='Overlap Risk Score', color='yellow')
plt.xlabel('Overlap')
plt.ylabel('Risk Score')
plt.title('Relationship Between Overlap and Risk Score')
plt.grid(True)
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()
