import time
import requests
import numpy as np # Using numpy makes percentile math bulletproof

# Ensure the URL exactly matches your running API configuration
url = "http://127.0.0.1:8000/watch/?user_id=196&title=Toy Story (1995)"
latencies = []

print("Benchmarking Two-Stage Production API Latency...")
for _ in range(100):
    start_time = time.perf_counter()
    response = requests.get(url)
    end_time = time.perf_counter()
    latencies.append((end_time - start_time) * 1000) # Convert to milliseconds

# Calculate clean, reliable metrics
avg_latency = sum(latencies) / 100
p95_latency = np.percentile(latencies, 95)

print(f"Average Latency: {avg_latency:.2f} ms")
print(f"P95 Latency (95th Percentile): {p95_latency:.2f} ms")

