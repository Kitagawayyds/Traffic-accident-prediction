import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Use default backend
matplotlib.use('TkAgg')  # Alternative: 'Qt5Agg'

def acceleration_risk(acceleration):
    """
    Calculate the risk score based on acceleration.

    Parameters:
    acceleration (float): The normalized acceleration value.

    Returns:
    float: The risk score for the given acceleration.
    """
    max_acceleration = 1  # Define maximum acceleration
    norm_acceleration = min(acceleration / max_acceleration, 1)  # Normalize acceleration
    risk_score = 10 * (norm_acceleration ** 2)  # Calculate risk score
    return risk_score

def angle_risk(angle):
    """
    Calculate the risk score based on angle change.

    Parameters:
    angle (float): The angle change in degrees.

    Returns:
    float: The risk score for the given angle change.
    """
    # Normalize angle to a range of 0 to 1
    normalized_angle = angle / 180.0

    # Apply cubic function for nonlinear risk mapping
    risk = normalized_angle ** 3

    # Map risk score to a range of 0 to 10
    risk_score = risk * 10

    return risk_score


def overlap_risk(overlap):
    """
    Maps the overlap degree to a risk score between 0 and 10 using a concave function with adjustable concavity.

    param overlap: Overlap degree, ranging from 0 to 1.
    param alpha: Parameter to adjust the concavity, default is 1.
    return: Risk score, ranging from 0 to 10.
    """
    # Adjust the log function for concavity
    risk_score = 10 * (np.log1p(overlap * 2) / np.log1p(2))  # Adjusted log function based on alpha

    return risk_score

# Create a range of values for acceleration from 0 to 1
acceleration_values = np.linspace(0, 1, 500)
acceleration_risk_scores = [acceleration_risk(a) for a in acceleration_values]

# Create a range of angle values from 0 to 180 degrees
angles = np.linspace(0, 180, 500)
angle_risk_scores = [angle_risk(angle) for angle in angles]

# Create a range of overlap values from 0 to 1
overlap_values = np.linspace(0, 1, 500)
risk_scores = [overlap_risk(overlap) for overlap in overlap_values]

# Plotting the risk scores

plt.figure(figsize=(18, 6))  # Set the figure size

# Plot acceleration risk score subplot
plt.subplot(1, 3, 1)
plt.plot(acceleration_values, acceleration_risk_scores, label='Acceleration Risk Score', color='blue')
plt.xlabel('Normalized Acceleration')
plt.ylabel('Risk Score')
plt.title('Acceleration to Risk Score Mapping')
plt.grid(True)
plt.legend()

# Plot angle risk score subplot
plt.subplot(1, 3, 2)
plt.plot(angles, angle_risk_scores, label='Angle Risk Score', color='red')
plt.xlabel('Angle Change (degrees)')
plt.ylabel('Risk Score')
plt.title('Angle Risk Score vs Angle Change')
plt.grid(True)
plt.legend()

# Plot overlap risk score subplot
plt.subplot(1, 3, 3)
plt.plot(overlap_values, risk_scores, label='Overlap Risk Score', color='yellow')
plt.xlabel('Overlap')
plt.ylabel('Risk Score')
plt.title('Relationship Between Overlap and Risk Score')
plt.grid(True)
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()
