import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
import psutil
import time
import multiprocessing as mp
import tempfile
import os
import threading

# ------------------ Worker that runs UMAP in a child process ------------------
def _umap_worker(X, n_neighbors, min_dist, out_path, queue):
    """Child process: run UMAP and save embedding to out_path (.npy)."""
    try:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        emb = reducer.fit_transform(X)
        np.save(out_path, emb)
        queue.put(("OK", None))
    except Exception as e:
        queue.put(("ERR", repr(e)))

# ------------------ Single function: simulate, run UMAP child, monitor timeline ------------------
def run_with_simulation_and_monitor(sim_kwargs, n_neighbors, min_dist, poll_interval=0.05, tmp_dir=None):
    """
    Simulate X,y (using sim_kwargs), monitor memory continuously while:
      - simulation runs (in main thread),
      - UMAP runs in a child process.
    Returns:
      embedding, elapsed_seconds, peak_proc_gb, peak_system_delta_gb, peak_cpu, mem_samples, sim_end_t, umap_start_t
    mem_samples is a list of (t_since_start, system_used_bytes, proc_mem_bytes_or_0, cpu_percent)
    sim_end_t and umap_start_t are seconds since monitoring start.
    """
    # Shared state for monitor thread
    stop_event = threading.Event()
    samples = []  # (t, system_used, proc_mem, cpu)
    proc_pid_container = {"pid": None}  # to be filled when UMAP child starts

    # monitoring thread polls system used memory and (if available) child proc private mem
    def monitor_loop():
        start_mon = time.time()
        # init CPU sample state
        psutil.cpu_percent(interval=None)
        while not stop_event.is_set():
            now = time.time() - start_mon
            system_used = psutil.virtual_memory().used
            proc_mem = 0
            pid = proc_pid_container.get("pid")
            if pid:
                try:
                    p = psutil.Process(pid)
                    try:
                        mf = p.memory_full_info()
                        proc_mem = getattr(mf, "uss", None) or getattr(mf, "rss", 0)
                    except Exception:
                        proc_mem = p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    proc_mem = 0
                except Exception:
                    proc_mem = 0
            # cpu_percent blocks for poll_interval (so we get a sensible sample)
            cpu = psutil.cpu_percent(interval=poll_interval)
            samples.append((now, system_used, proc_mem, cpu))
        # final sample after stop (non-blocking)
        now = time.time() - start_mon
        system_used = psutil.virtual_memory().used
        pid = proc_pid_container.get("pid")
        proc_mem = 0
        if pid:
            try:
                p = psutil.Process(pid)
                try:
                    mf = p.memory_full_info()
                    proc_mem = getattr(mf, "uss", None) or getattr(mf, "rss", 0)
                except Exception:
                    proc_mem = p.memory_info().rss
            except Exception:
                proc_mem = 0
        cpu = psutil.cpu_percent(interval=None)
        samples.append((now, system_used, proc_mem, cpu))

    # Start monitoring before simulation
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    mon_start_time = time.time()
    try:
        # pick last sampled system_used if samples exist
        baseline_system_used = samples[-1][1] if samples else psutil.virtual_memory().used
    except Exception:
        baseline_system_used = 0
    # Pause briefly to ensure monitor thread is running and has taken an initial sample
    time.sleep(1)
    # --- Simulation (this is happening while monitor is running) ---
    X, y = make_blobs(**sim_kwargs)
    sim_end_t = time.time() - mon_start_time
    time.sleep(1)  # small pause to ensure monitor thread samples after sim end
    # Baseline for system delta (use value just before starting UMAP)


    # ----------------- Start UMAP child -----------------
    q = mp.Queue()
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    fd, out_path = tempfile.mkstemp(suffix=".npy", dir=tmp_dir)
    os.close(fd)
    os.remove(out_path)

    proc = mp.Process(target=_umap_worker, args=(X, n_neighbors, min_dist, out_path, q))
    umap_start_abs = time.time()
    proc.start()
    proc_pid_container["pid"] = proc.pid  # tell monitor to look at this pid
    umap_start_t = umap_start_abs - mon_start_time

    # wait for child to finish while monitor thread keeps polling
    try:
        proc.join()
    except Exception:
        proc.terminate()
        proc.join()

    # Stop monitoring
    stop_event.set()
    monitor_thread.join()

    elapsed = time.time() - mon_start_time

    # Check child status from queue
    status, err = ("ERR", "unknown")
    try:
        if not q.empty():
            status, err = q.get_nowait()
    except Exception:
        pass

    if status != "OK":
        raise RuntimeError(f"Child UMAP process failed: {err}")

    if not os.path.exists(out_path):
        raise RuntimeError("Embedding file not found after child finished.")
    embedding = np.load(out_path)
    os.remove(out_path)

    # Analyze samples: find peaks
    peak_proc = 0
    peak_system_delta = 0
    peak_cpu = 0.0
    for (_, system_used, proc_mem, cpu) in samples:
        peak_proc = max(peak_proc, proc_mem)
        peak_system_delta = max(peak_system_delta, max(0, system_used - baseline_system_used))
        peak_cpu = max(peak_cpu, cpu)

    peak_proc_gb = peak_proc / (1024 ** 3)
    peak_system_delta_gb = peak_system_delta / (1024 ** 3)

    return embedding, elapsed, peak_proc_gb, peak_system_delta_gb, peak_cpu, samples, sim_end_t, umap_start_t

# ------------------ Main run_umap (uses the above) ------------------
def run_umap(n_samples=None, n_features=None, n_clusters=None, n_neighbors_list=None, min_dist_list=None):
    # Simulate defaults:
    if n_samples is None:
        n_samples = 1000
    if n_features is None:
        n_features = 50
    if n_clusters is None:
        n_clusters = 5

    # Prepare simulation kwargs to pass to make_blobs
    sim_kwargs = {"n_samples": n_samples, "n_features": n_features, "centers": n_clusters, "random_state": 42}

    # Compute pairwise distances in the original space will be computed after we have X,
    # so we will compute original_distances inside the measurement function if needed.
    if n_neighbors_list is None:
        n_neighbors_list = [30]
    if min_dist_list is None:
        min_dist_list = [0.1]

    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            print(f"Testing UMAP with n_neighbors={n_neighbors} and min_dist={min_dist} (n_samples={n_samples}, n_features={n_features})...")
            # Run simulation + UMAP with continuous monitoring
            embedding, elapsed, peak_proc_gb, peak_system_delta_gb, peak_cpu, mem_samples, sim_end_t, umap_start_t = run_with_simulation_and_monitor(
                sim_kwargs, n_neighbors, min_dist, poll_interval=0.05
            )

            print(f"Elapsed total (simulate + UMAP): {elapsed:.3f} s")
            print(f"Peak process private mem: {peak_proc_gb:.3f} GB, Peak system used Δ: {peak_system_delta_gb:.3f} GB, Peak CPU sampled: {peak_cpu:.1f}%")

            # Now compute original distances and correlation (we have X from the worker? we don't — so recompute X for distances here)
            # Recreate X,y to compute original distances exactly the same way (cheap compared to earlier steps)
            X_check, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
            original_distances = pairwise_distances(X_check)
            embedded_distances = pairwise_distances(embedding)
            correlation = np.corrcoef(original_distances.flatten(), embedded_distances.flatten())[0, 1]
            print(f"n_neighbors: {n_neighbors}, min_dist: {min_dist}, distance correlation: {correlation:.4f}")

            # Plot embedding (same as before):
            plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='viridis', s=10)
            plt.xlabel(f"n_neighbors={n_neighbors}, min_dist={min_dist}, time={elapsed:.2f}s")
            plt.ylabel("UMAP Dimension 2")
            plt.title(f"Sample size used: {n_samples}, Features used: {n_features}, distance correlation: {correlation:.4f}")

            plt.text(0.5, 0.95, f"Peak proc RAM: {peak_proc_gb:.2f} GB\nSystem Δ: {peak_system_delta_gb:.2f} GB\nPeak CPU: {peak_cpu:.0f}%", transform=plt.gca().transAxes, ha='center', va='center')
            plt.savefig(f"umap_n_neighbors_{n_neighbors}_min_dist_{min_dist}_samplesize_{n_samples}_features_{n_features}.png")
            plt.close()

            # --- Plot memory timeline including simulation and UMAP start marker ---
            if mem_samples:
                times = [s[0] for s in mem_samples]
                system_used_vals = [s[1] / (1024 ** 3) for s in mem_samples]  # GB
                proc_mem_vals = [s[2] / (1024 ** 3) for s in mem_samples]     # GB
                cpu_vals = [s[3] for s in mem_samples]

                plt.figure(figsize=(10, 4))
                plt.plot(times, system_used_vals, label='System used (GB)')
                plt.plot(times, proc_mem_vals, label='Child proc private (GB)')
                # vertical line showing when UMAP started
                plt.axvline(x=umap_start_t, color='red', linestyle='--', label='UMAP start')
                # vertical line showing when simulation finished
                plt.axvline(x=sim_end_t, color='gray', linestyle=':', label='Simulation end')
                plt.xlabel('Seconds since monitor start')
                plt.ylabel('Memory (GB)')
                plt.title(f"Memory timeline: n_neighbors={n_neighbors}, min_dist={min_dist}, samples={n_samples}, features={n_features}")
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(f"memory_timeline_n_neighbors_{n_neighbors}_min_dist_{min_dist}_samplesize_{n_samples}_features_{n_features}.png")
                plt.close()

def test_umap_with_diff_sizes():
    # Test UMAP with different sizes of data:
    for n_samples in [10000, 20000,25000]:
        for n_features in [500]:
            print(f"Testing UMAP with n_samples={n_samples} and n_features={n_features}...")
            # call run_umap with these params
            run_umap(n_samples=n_samples, n_features=n_features)

if __name__ == '__main__':
    test_umap_with_diff_sizes()
