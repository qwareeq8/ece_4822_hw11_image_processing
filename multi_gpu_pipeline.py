#!/usr/bin/env python
#
# file: multi_gpu_pipeline.py
#
# revision history:
# 20241215 (YQ): first version
#
# description:
#  This Python script implements a multi-threaded, CUDA-accelerated image
#  pipeline involving Gaussian blur, Fourier transform, and histogram
#  calculations. The script manages reading image windows using
#  image_parse_utils.get_frames, then processing those windows on multiple GPUs.
#

import os
import threading
import numpy as np
import cupy as cp
from numba import cuda

import image_parse_utils

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

__FILE__ = os.path.basename(__file__)

MATRIX_THREAD_COUNT = 8
MATRIX_BLOCK_COUNT = 32
WINDOW_THREAD_COUNT = 12
WINDOW_BLOCK_COUNT = 250
WINDOWS_PER_ROUND = WINDOW_THREAD_COUNT * WINDOW_BLOCK_COUNT

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#------------------------------------------------------------------------------
#
# CUDA kernels
#
#------------------------------------------------------------------------------

@cuda.jit
def gaussian_blur(src_arr, dst_arr, kernel_arr):
    """
    function: gaussian_blur

    arguments:
     input_array: the input windows on which Gaussian blur is applied
     blurred_array: output array storing blurred result
     blur_kernel: the kernel for Gaussian convolution

    return:
     none

    description:
     This CUDA kernel applies a 2D Gaussian blur to each window in input_array.
     The kernel loops over channels, computing weighted sums using blur_kernel.
    """
    gx, gy, gz = cuda.grid(3)
    kern_h, kern_w = kernel_arr.shape
    win_count, h_val, w_val, chan_count = src_arr.shape
    rad = kern_h // 2
    for c_idx in range(chan_count):
        for ky in range(-rad, rad + 1):
            for kx in range(-rad, rad + 1):
                if (gx >= rad and gy >= rad
                    and (h_val - gx) > rad
                    and (w_val - gy) > rad):
                    dst_arr[gz, gx, gy, c_idx] += (
                        src_arr[gz, gx + ky, gy + kx, c_idx]
                        * kernel_arr[ky + rad, kx + rad]
                    )

@cuda.jit
def histogram_gpu(src_arr, out_hist):
    """
    function: histogram_gpu

    arguments:
     input_array: GPU array of shape (windows, height, width, channels)
     output: a (channels, 16) GPU array storing histogram counts

    return:
     none

    description:
     This CUDA kernel performs histogram binning across each channel.
     Each bin covers a specific intensity range. Accumulates counts
     in 'output'. The intensity ranges were calculated by finiding
     the global and local minima across the entire dataset.
    """
    gx, gy, gz = cuda.grid(3)
    w_count, h_val, w_val, chan_count = src_arr.shape
    for c_idx in range(chan_count):
        val = src_arr[gz, gx, gy, c_idx]
        if val < 836319:
            out_hist[c_idx, 0] += 1
        elif val < 1672638:
            out_hist[c_idx, 1] += 1
        elif val < 2508957:
            out_hist[c_idx, 2] += 1
        elif val < 3345276:
            out_hist[c_idx, 3] += 1
        elif val < 4181595:
            out_hist[c_idx, 4] += 1
        elif val < 5017914:
            out_hist[c_idx, 5] += 1
        elif val < 5854233:
            out_hist[c_idx, 6] += 1
        elif val < 6690552:
            out_hist[c_idx, 7] += 1
        elif val < 7526871:
            out_hist[c_idx, 8] += 1
        elif val < 8363190:
            out_hist[c_idx, 9] += 1
        elif val < 9199509:
            out_hist[c_idx, 10] += 1
        elif val < 10035828:
            out_hist[c_idx, 11] += 1
        elif val < 10872147:
            out_hist[c_idx, 12] += 1
        elif val < 11708466:
            out_hist[c_idx, 13] += 1
        elif val < 12544785:
            out_hist[c_idx, 14] += 1
        else:
            out_hist[c_idx, 15] += 1

#------------------------------------------------------------------------------
#
# other supporting functions
#
#------------------------------------------------------------------------------

def create_blur_kernel(k_rad):
    """
    function: generate_blur_kernel

    arguments:
     kernel_radius: integer radius for the Gaussian kernel

    return:
     blur_kernel: a 2D NumPy array containing a normalized Gaussian kernel

    description:
     This function generates a (2 * kernel_radius + 1) x (2 * kernel_radius + 1)
     Gaussian kernel. The sigma is approximated as max(kernel_radius/2, 1).
    """
    sigma_v = max(float(k_rad / 2.0), 1.0)
    k_width = 2 * k_rad + 1
    filter_mat = np.zeros((k_width, k_width), dtype=np.float32)
    accum = 0.0
    for i in range(-k_rad, k_rad + 1):
        for j in range(-k_rad, k_rad + 1):
            exp_num = -1.0 * (i * i + j * j)
            exp_den = 2.0 * sigma_v * sigma_v
            e_expr = np.exp(exp_num / exp_den)
            val_kern = e_expr / (2.0 * np.pi * sigma_v * sigma_v)
            filter_mat[i + k_rad, j + k_rad] = val_kern
            accum += val_kern
    filter_mat /= accum
    return filter_mat

def process_batch(sub_data, blur_obj, device_idx, shared_mem):
    """
    function: process_batch

    arguments:
     subset: array of shape (batch_size, height, width, channels)
     kernel: a 2D Gaussian kernel (NumPy array)
     dev_id: integer specifying which GPU device to use
     shared_output: a shared list storing results for each device

    return:
     none

    description:
     This function transfers the windows to GPU memory, applies Gaussian
     blur, computes FFT and magnitude, then accumulates histogram results
     using histogram_gpu. The histogram result is placed in mutable_output[device_id].
    """
    with cp.cuda.Device(device_idx):
        cuda.select_device(device_idx)
        kern_gpu = cp.array(blur_obj, dtype=cp.float32)
        arr_win = cp.array(sub_data, dtype=cp.uint8)
        arr_blur = cp.zeros_like(arr_win, dtype=cp.uint8)

        gaussian_blur[(MATRIX_BLOCK_COUNT, MATRIX_BLOCK_COUNT, WINDOW_BLOCK_COUNT),
                      (MATRIX_THREAD_COUNT, MATRIX_THREAD_COUNT, WINDOW_THREAD_COUNT)](
            arr_win, arr_blur, kern_gpu
        )

        fft_calc = cp.fft.fft2(arr_blur, axes=(1,2))
        magnitude_vals = cp.abs(fft_calc)

        local_res = cp.zeros((3,16), dtype=cp.int32)
        histogram_gpu[(MATRIX_BLOCK_COUNT, MATRIX_BLOCK_COUNT, WINDOW_BLOCK_COUNT),
                      (MATRIX_THREAD_COUNT, MATRIX_THREAD_COUNT, WINDOW_THREAD_COUNT)](
            magnitude_vals, local_res
        )
        shared_mem[device_idx] = local_res.get()

def main():
    """
    function: main

    arguments:
     none

    return:
     none

    description:
     This is the main entry point. It reads a list of image paths, loads
     them using image_parse_utils.get_frames, processes them in batches
     on multiple GPUs, and accumulates histogram results in a NumPy array.
    """
    with open('./images.list', 'r') as fp:
        svs_paths = [p.strip() for p in fp if p.strip()]

    print("SVS Paths to process:", svs_paths, flush=True)
    container_output = [None]
    initial_path = svs_paths.pop(0)
    loader_thread = threading.Thread(target=image_parse_utils.get_frames, args=(initial_path, container_output, 0, 4000))
    loader_thread.start()

    blur_kern = create_blur_kernel(3)
    final_hist = np.zeros((3,16), dtype=int)
    dev_out = [0,0,0,0]

    for idx, img_pth in enumerate(svs_paths):
        loader_thread.join()
        print(f"Image {idx + 1} loaded", flush=True)

        loader_thread = threading.Thread(target=image_parse_utils.get_frames, args=(img_pth, container_output, 0, 4000))
        loader_thread.start()

        data_wins = container_output[0]
        for st_idx in range(0, len(data_wins) - (4 * WINDOWS_PER_ROUND), 4 * WINDOWS_PER_ROUND):
            t_list = []
            for d_id in range(4):
                begin = st_idx + d_id*WINDOWS_PER_ROUND
                end = begin + WINDOWS_PER_ROUND
                subset = data_wins[begin:end]
                th = threading.Thread(target=process_batch, args=(subset, blur_kern, d_id, dev_out))
                th.start()
                t_list.append(th)

            for th in t_list:
                th.join()

            for hist_res in dev_out:
                final_hist += hist_res
            print(f"{st_idx + 4*WINDOWS_PER_ROUND} / {len(data_wins)} processed", flush=True)

    loader_thread.join()
    leftover_data = container_output[0]
    for st_idx in range(0, len(leftover_data) - (4 * WINDOWS_PER_ROUND), 4 * WINDOWS_PER_ROUND):
        batch_threads = []
        for d_id in range(4):
            b_start = st_idx + d_id*WINDOWS_PER_ROUND
            b_end = b_start + WINDOWS_PER_ROUND
            sub_block = leftover_data[b_start:b_end]
            new_thr = threading.Thread(target=process_batch, args=(sub_block, blur_kern, d_id, dev_out))
            new_thr.start()
            batch_threads.append(new_thr)

        for new_thr in batch_threads:
            new_thr.join()

        for hist_res in dev_out:
            final_hist += hist_res
        print(f"{st_idx + 4*WINDOWS_PER_ROUND} / {len(leftover_data)} processed", flush=True)

    print("Final aggregated histogram:\n", final_hist, flush=True)
    np.save('final_histogram.npy', final_hist)

#------------------------------------------------------------------------------
#
# entry point
#
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
