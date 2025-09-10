import os
import time
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import pathwaysutils


def benchmark_host_to_device_throughput():
  """Benchmarks JAX device_put throughput from CPU host to a v5e-32 TPU slice.
  """
  print(f"JAX version: {jax.__version__}")
  devices = jax.devices() if os.environ.get("JAX_PLATFORMS") else jax.local_devices()
  num_devices = len(devices)
  print(f"Available devices: {num_devices}")

  data_bytes_per_device = 1 * 1024 * 1024 * 1024  # 1 GiB
  dtype = np.float32
  num_elements = data_bytes_per_device // np.dtype(dtype).itemsize
  data_gb_per_device = num_elements * np.dtype(dtype).itemsize / (1024**3)

  print(
      f"Creating a NumPy array of shape ({num_elements},) type {dtype}, size"
      f" {data_gb_per_device:.2f} GiB"
  )
  host_array = np.arange(num_elements, dtype=dtype)

  # Create a mesh spanning all devices.
  mesh = Mesh(np.array(devices), axis_names=("i",))
  # An empty PartitionSpec() means the array is fully replicated across all
  # devices in the mesh.
  replicated_sharding = NamedSharding(mesh, PartitionSpec())
  print(f"Using sharding for replication: {replicated_sharding}")

  # Warm-up transfer
  print("Performing warm-up transfer...")
  try:
    dummy_array = jax.device_put(host_array, replicated_sharding)
    dummy_array.block_until_ready()
    print("Warm-up complete.")
  except Exception as e:
    print(f"Error during warm-up: {e}")
    return

  # Benchmark loop
  num_transfers = 5
  transfer_times = []

  print(f"Starting benchmark ({num_transfers} transfers)...")
  for i in range(num_transfers):
    if i == 0:
      jax.profiler.start_trace("gs://cloud-tpu-multipod-dev-axlearn/traces/proxy-benchmark")
    start_time = time.perf_counter()
    device_array = jax.device_put(host_array, replicated_sharding)
    device_array.block_until_ready()
    end_time = time.perf_counter()

    duration = end_time - start_time
    transfer_times.append(duration)
    print(f"Transfer {i+1}/{num_transfers}: {duration:.4f} seconds")
    if i == 0:
      jax.profiler.stop_trace()
    del device_array  # Optional: hint for early deletion

  avg_time = np.mean(transfer_times)
  print(f"\nAverage time per device_put call: {avg_time:.4f} seconds")

  total_data_moved_gb = data_gb_per_device * num_devices
  throughput_gb_s = total_data_moved_gb / avg_time

  print(f"Data per device: {data_gb_per_device:.2f} GiB")
  print(
      "Total data transferred from host per operation:"
      f" {total_data_moved_gb:.2f} GiB"
  )
  print(f"Aggregated Host -> Devices Throughput: {throughput_gb_s:.2f} GiB/s")
  print(f"Aggregated Host -> Devices Throughput: {throughput_gb_s * 8:.2f} Gbps/s")


if __name__ == "__main__":
  pathwaysutils.initialize() if os.environ.get("JAX_PLATFORMS") =="proxy" else jax.distributed.initialize()
  benchmark_host_to_device_throughput()
