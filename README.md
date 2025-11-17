# ## Smart-Energy-Optimizer

Creating a complex system like the Smart-Energy-Optimizer involves multiple components, including data collection from IoT devices, data processing, and machine learning for optimization. Below is a simplified version of such a system in Python. The program demonstrates a basic outline that can be expanded upon. It simulates energy consumption data, trains a simple machine learning model to predict future consumption, and provides recommendations for optimizing energy use.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# A mock function to simulate data collection from IoT sensors
def collect_energy_data(num_samples):
    """
    Simulate the collection of energy consumption data from IoT devices.
    Args:
    - num_samples: Number of data samples to generate.

    Returns:
    - DataFrame with simulated timestamps and energy consumption values.
    """
    # Simulate timestamps and energy usage data
    timestamps = pd.date_range(start='2023-10-01', periods=num_samples, freq='H')
    energy_usage = np.random.normal(loc=100, scale=20, size=num_samples)  # Random energy usage
    return pd.DataFrame({'timestamp': timestamps, 'energy_usage': energy_usage})

# Simulate data collection
try:
    data = collect_energy_data(1000)
except Exception as e:
    print("Error during data collection:", e)

# Basic visualization of energy usage data
try:
    plt.figure(figsize=(14, 6))
    plt.plot(data['timestamp'], data['energy_usage'], label='Energy Usage')
    plt.xlabel('Timestamp')
    plt.ylabel('Energy Usage (kWh)')
    plt.title('Energy Usage Over Time')
    plt.legend()
    plt.show()
except Exception as e:
    print("Error during data visualization:", e)

# Prepare the data for machine learning
try:
    # Feature engineering: extract hour from the timestamp as a simple feature
    data['hour'] = data['timestamp'].dt.hour
    features = data[['hour']]
    target = data['energy_usage']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
except Exception as e:
    print("Error during data preparation for ML:", e)

# Train a simple machine learning model
try:
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Predict on the test data
    predictions = model.predict(X_test)
    # Evaluate the model
    mse = np.mean((predictions - y_test)**2)
    print(f"Model Mean Squared Error: {mse:.2f}")
except Exception as e:
    print("Error during model training or evaluation:", e)

# Simple function to recommend optimization strategies
def recommend_optimizations(energy_usage, baseline=100):
    """
    Provide simple suggestions to optimize energy based on usage.
    Args:
    - energy_usage: The current energy usage.
    - baseline: The baseline average usage.

    Returns:
    A string with recommendations.
    """
    try:
        if energy_usage > baseline:
            return "Consider reducing energy usage during peak hours."
        else:
            return "Energy usage is optimal."
    except Exception as e:
        return f"Error in generating recommendations: {e}"

# Apply recommendations on test data
try:
    for val in predictions[:10]:  # Just taking 10 samples
        print(f"Predicted Usage: {val:.2f} kWh, Recommendation: {recommend_optimizations(val)}")
except Exception as e:
    print("Error during recommendation generation:", e)
```

### Explanation:

1. **Data Collection**: We simulate energy consumption data using a mock function. In reality, this would involve collecting data from IoT devices.

2. **Data Visualization**: Using `matplotlib` to plot energy usage over time, which provides insights into consumption patterns.

3. **Data Preparation**: Extracting features (hour of the day, in this case) and preparing the data for machine learning by splitting it into training and test sets.

4. **Modeling**: Training a simple linear regression model to predict energy usage.

5. **Recommendations**: Based on predicted energy usage, the system provides optimization suggestions.

6. **Error Handling**: Try-except blocks are used throughout to handle potential errors gracefully.

This is a simplified prototype. A real system would include additional data features, more sophisticated models, integration with IoT platforms, a user interface, and could be extended with additional functionalities for real-world application.