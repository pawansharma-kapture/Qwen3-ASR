# PyTorch DataLoader Parameters Explanation

This document explains the configuration parameters for the PyTorch `DataLoader` used in `finetuning/config.yaml`. These settings control how data is loaded from disk/RAM into the model training loop, directly impacting training speed and GPU utilization.

## 1. `num_workers: 8`

*   **What it does:** Sets the number of subprocesses used for data loading.
    *   `0`: The main process loads the data (synchronous). Training pauses while data is loaded.
    *   `>0`: Multiple worker processes load data in parallel (asynchronous). The main process (training loop) can consume a batch while workers prepare the next ones.
*   **Effect:**
    *   **Pros:** Significantly speeds up data loading, preventing the GPU from waiting (starvation).
    *   **Cons:** Increases system RAM usage (each worker needs memory) and CPU overhead.
*   **Recommendation:** A good rule of thumb is `4 * number_of_gpus` or equal to the number of CPU physical cores, but not exceeding system RAM limits. If you see warnings about "shared memory," reduce this or increase shared memory size (e.g., in Docker).

## 2. `pin_memory: 1` (True)

*   **What it does:** If `True`, the data loader copies tensors into CUDA pinned memory (page-locked memory) before returning them.
*   **Effect:**
    *   **Pros:** Enables much faster data transfer from CPU (Host) to GPU (Device).
    *   **Cons:** Uses a bit more host RAM.
*   **Recommendation:** Almost always set to `True` when training on NVIDIA GPUs. Set to `False` only if you are extremely constrained on CPU RAM or training on CPU only.

## 3. `persistent_workers: 1` (True)

*   **What it does:** If `True`, the worker processes are kept alive after a dataset has been consumed once (i.e., after an epoch ends). If `False`, workers are killed and re-spawned every epoch.
*   **Effect:**
    *   **Pros:** Saves the overhead of starting up new processes at the beginning of every epoch. This is crucial if your dataset is small or epochs are short.
    *   **Cons:** Keeps worker memory allocated even when not strictly "loading" (e.g., during validation if validation uses a different loader).
*   **Recommendation:** Set to `True` to reduce latency between epochs.

## 4. `prefetch_factor: 2`

*   **What it does:** Controls how many batches *each worker* loads in advance.
    *   Total prefetched batches = `num_workers * prefetch_factor`.
*   **Effect:**
    *   **Pros:** Creates a buffer of ready data. If one batch takes unusually long to process (e.g., complex augmentation), the buffer prevents the GPU from waiting.
    *   **Cons:** Increases memory usage linearly. If `num_workers` is high, a high `prefetch_factor` can quickly exhaust RAM.
*   **Recommendation:** `2` is a standard default. Increase to `4` or `8` if data loading is the bottleneck (GPU utilization < 100%) and you have plenty of RAM. Decrease if you run out of memory.

---

## Summary of Trade-offs

| Parameter | Increase leads to... | Decrease leads to... |
| :--- | :--- | :--- |
| **`num_workers`** | Faster loading, higher RAM usage | Slower loading, lower RAM usage |
| **`pin_memory`** | Faster CPU->GPU transfer | Slower transfer |
| **`persistent_workers`** | Faster epoch transitions | Lower idle RAM usage |
| **`prefetch_factor`** | Smoother data stream, higher RAM usage | Risk of GPU starvation |
