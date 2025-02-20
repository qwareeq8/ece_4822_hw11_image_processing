# ECE 4822 HW11: Image Processing – Python Implementation

This repository contains the Python source code for a CUDA-accelerated, multi-threaded image processing pipeline. The solution is organized into two files:

- **multi_gpu_pipeline.py** – Implements the main pipeline: it loads image windows, applies a 2D Gaussian blur via a custom CUDA kernel, computes a 2D FFT (with magnitude extraction), and bins the results into a histogram using another CUDA kernel. The workload is distributed across multiple GPUs using Python threading.
- **image_parse_utils.py** – Provides image parsing functions that leverage nedc_image_tools to open images, generate subwindow coordinates, and extract subwindows based on a sliding window strategy.

## Approach
The solution is designed to efficiently process large image databases by:
- **Multi-threading:** A separate thread loads image frames while other threads process batches of image windows concurrently.
- **GPU Acceleration:** Two custom CUDA kernels are used – one for applying a 7×7 Gaussian blur and another for computing histograms from the FFT magnitude values.
- **Batch Processing:** Images are partitioned into manageable batches and processed in parallel across four GPUs.
- **Data Flow:** Image windows are extracted via a sliding window (256×256 windows with a 128×128 shift) using functions in *image_parse_utils.py*. These windows are then blurred, transformed via FFT, and their spectral magnitudes are histogrammed and aggregated.

## Getting Started
1. **Environment:** Ensure Python, CUDA, CuPy, and Numba are installed.
2. **Image List:** Place your image paths in a file named `images.list`.
3. **Run the Pipeline:**
   ```bash
   python multi_gpu_pipeline.py
   ```
4. **Output:** The final aggregated histogram is saved as `final_histogram.npy`.

## Additional Information
- **Assignment Specification:**  
  [HW11 Specification](https://isip.piconepress.com/courses/temple/ece_4822/homework/2024_01_fall/hw_11_v00.docx)
- **Benchmark Results:**  
  [Current Benchmarks](https://isip.piconepress.com/courses/temple/ece_4822/resources/benchmarks/image_processing/current/)
