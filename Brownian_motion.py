import numpy as np
import matplotlib.pyplot as plt

import requests

url = "https://api.open-meteo.com/v1/forecast"

params = {
"latitude": 51.51,
    "longitude": -0.13,
    "hourly": "wind_speed_10m"

}

data = requests.get(url, params=params, timeout=10)
data.raise_for_status()

wind = np.array(data.json()["hourly"]["wind_speed_10m"])

# Simulation parameters
particles = 10000
steps = 1000

# Generate random directions - Gaussian-distributed dx and dy
angles = np.random.normal(0, 2*np.pi, (particles, steps))

mu = 0.2 #drift
sigma = 0.5 + 0.05 * np.resize(wind, steps) #volatility 
dt = 1.0 #time step

time = np.arange(steps)

sigma = 1.0 + 0.5 * np.sin(time * 0.1)

# Calculate movement

dx = mu * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (particles, steps))
dy = sigma * np.sqrt(dt) * np.random.normal(0, 1,(particles, steps))

# Accumulate positions
x = np.cumsum(dx, axis=1)
y = np.cumsum(dy, axis=1)

# Visualise the trajectories
plt.figure(figsize=(8, 8))

for i in range(30):
    plt.plot(x[i], y[i], alpha=0.3)

# Visualise final positions
plt.scatter(x[:, -1], y[:, -1], s=5)

plt.scatter(0, 0, color="red", label="Origin")
plt.axis("equal")
plt.legend()
plt.title("Brownian-motion-Simulation-using-Open-Meteo-API-")
plt.show()
