# Datasets

### Steam Games Requirements (raw/steam_games_requirements.csv) by HuggingFace

* https://huggingface.co/datasets/swamysharavana/steam_games.csv
- Records: ~40,000
- Description: Steam games with system requirements (CPU, RAM, GPU)

### CPU UserBenchmarks (raw/CPU_UserBenchmarks.csv) by UserBenchmark.com

* https://www.userbenchmark.com/resources/download/csv/CPU_UserBenchmarks.csv
- Records: 1,423
- Description: CPU benchmark scores

### GPU Benchmarks v7 (raw/GPU_benchmarks_v7.csv) by Kaggle/alanjo

* https://www.kaggle.com/datasets/alanjo/gpu-benchmarks/
- Records: 2,317
- Description: GPU benchmark scores

### AMD Processors (raw/AMD.csv) by Kaggle/alanjo

* https://www.kaggle.com/datasets/alanjo/amd-processor-specifications/
- Records: 582
- Description: AMD processor specifications

### Intel Processors (raw/INTEL.csv) by Kaggle/alanjo

* https://www.kaggle.com/datasets/alanjo/amd-processor-specifications/
- Records: 2,880
- Description: Intel processor specifications

### Video Game Requirements v2 (../pc_requirements.csv) by Kaggle/baraazaid

* https://www.kaggle.com/datasets/baraazaid/pc-video-game-requirements-v2
- Records: 10,849
- Description: PC video game requirements

### CPU Benchmarks Compilation (raw/CPU_benchmark_v4.csv) by Kaggle/alanjo

* https://www.kaggle.com/datasets/alanjo/cpu-benchmarks
- Records: 3,825
- Description: CPU benchmarks compilation

### GPU Specs (raw/gpu.csv) by Kaggle/kkhandekar

* https://www.kaggle.com/datasets/kkhandekar/cpu-gpu-specs
- Records: 250
- Description: Popular GPU specs

### GPU Scores Graphics APIs (raw/GPU_scores_graphicsAPIs.csv) by Kaggle/alanjo

* https://www.kaggle.com/datasets/alanjo/gpu-benchmarks/
- Records: 1,213
- Description: GPU scores by graphics API

---

# Data Cleaning/Compatibility

### Cleaning rules

* Desktop hardware only

## Data Modeling

### CPUs

* AMD Supported Sockets: AM3/AM4/AM5

* Intel Supported Sockets: 1150/1151/1200/1700/1851

### GPUs

CPU Datasets with clockspeed, number of cores/threads
* **Features:** processor, year, cores, threads, baseClock, maxboostClock, L1, L2, L3(Caches), memType, memSpec, Socket

* **TODO: Scraping to get updated hardware prices for best results**

* **TODO: Adopt LightGBM and RSO for k-means optimization**
