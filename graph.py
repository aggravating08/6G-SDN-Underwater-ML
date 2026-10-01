import matplotlib.pyplot as plt

# execution times

implementations = [
    "Serial",
    "OpenMP",
    "MPI-2",
    "MPI-4"
]

times = [
    2.49275,
    1.40972,
    1.72584,
    1.72516
]

# speedups

serial_time = 2.49275

speedups = [
    serial_time / 2.49275,
    serial_time / 1.40972,
    serial_time / 1.72584,
    serial_time / 1.72516
]

# -----------------------------
# Runtime Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.bar(implementations, times)

plt.xlabel("Implementation")
plt.ylabel("Execution Time (sec)")
plt.title("Laplace Solver Runtime Comparison")

plt.savefig("runtime_graph.png")

# -----------------------------
# Speedup Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.bar(implementations, speedups)

plt.xlabel("Implementation")
plt.ylabel("Speedup")
plt.title("Parallel Speedup Comparison")

plt.savefig("speedup_graph.png")

print("Graphs generated successfully.")
