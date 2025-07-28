import os
import matplotlib.pyplot as plt
import numpy as np

def extract_throughput_mean(filepath):
    with open(filepath, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return np.mean(values) if values else None

def walk_and_plot(root_dir, plot_root="plots_host_new"):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not filenames:
            continue  # skip empty dirs

        # Get relative path from root_dir
        rel_path = os.path.relpath(dirpath, root_dir)  # e.g. copy/device

        benchmark_name = os.path.basename(dirpath)  # e.g. device
        x_nbytes = []
        y_throughput = []

        for fname in sorted(filenames):
            if fname.isdigit():
                n_bytes = int(fname)
                filepath = os.path.join(dirpath, fname)
                mean_throughput = extract_throughput_mean(filepath)
                if mean_throughput is not None:
                    x_nbytes.append(n_bytes)
                    y_throughput.append(mean_throughput)
            else:
                print(f"Skipping non-numeric file: {fname}")

        if x_nbytes and y_throughput:
            print(f"Saving plot for benchmark: {rel_path}")
            x_nbytes, y_throughput = zip(*sorted(zip(x_nbytes, y_throughput)))

            plt.figure()
            plt.plot(x_nbytes, y_throughput, marker='o')
            plt.xscale('log')
            plt.xlabel("n_bytes")
            plt.ylabel("Avg Throughput (GB/s)")
            plt.title(f"Benchmark: {rel_path}")
            plt.grid(True)
            plt.tight_layout()

            # Create corresponding directory inside plots/
            save_dir = os.path.join(plot_root, os.path.dirname(rel_path))
            os.makedirs(save_dir, exist_ok=True)

            # Save with benchmark_name.png inside the save_dir
            output_path = os.path.join(save_dir, f"{benchmark_name}.png")
            plt.savefig(output_path)
            plt.close()
        else:
            print(f"No valid data in: {rel_path}")

if __name__ == "__main__":
    benchmark_root = "results"  # your root with benchmark data
    walk_and_plot(benchmark_root)

