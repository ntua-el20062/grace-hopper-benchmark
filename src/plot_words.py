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
            continue  

        rel_path = os.path.relpath(dirpath, root_dir)
        benchmark_name = os.path.basename(dirpath)

        x_labels = []
        y_throughput = []

        for fname in sorted(filenames):  
            if not fname.isdigit():  
                filepath = os.path.join(dirpath, fname)
                mean_throughput = extract_throughput_mean(filepath)
                if mean_throughput is not None:
                    x_labels.append(fname)
                    y_throughput.append(mean_throughput)
            else:
                print(f"Skipping numeric file: {fname}")

        if x_labels and y_throughput:
            print(f"Saving plot for benchmark: {rel_path}")

            plt.figure()
            plt.plot(x_labels, y_throughput, marker='o')
            plt.xlabel("Configuration")
            plt.ylabel("Avg Throughput (GB/s)")
            plt.title(f"Benchmark: {rel_path}")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()

            save_dir = os.path.join(plot_root, os.path.dirname(rel_path))
            os.makedirs(save_dir, exist_ok=True)

            output_path = os.path.join(save_dir, f"{benchmark_name}.png")
            plt.savefig(output_path)
            plt.close()
        else:
            print(f"No non-numeric data in: {rel_path}")

if __name__ == "__main__":
    benchmark_root = "results"  
    walk_and_plot(benchmark_root)
