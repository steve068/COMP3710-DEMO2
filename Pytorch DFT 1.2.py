import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import time

# 检查是否有可用的 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set parameters for the signal
N = 8192 # Number of sample points
T = 1.0 # Duration of the signal in seconds
f0 = 1 # Fundamental frequency of the square wave in Hz
# List of harmonic numbers used to construct the square wave
harmonics = [1, 3, 5]
# Define the square wave function
def square_wave_pytorch(t_tensor, f0):
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t_tensor))
# Fourier series approximation of the square wave
def square_wave_fourier_pytorch(t_tensor, f0, N_harmonics):
    result = torch.zeros_like(t_tensor)
    for k in range(N_harmonics):
        n = 2 * k + 1
        result += torch.sin(2 * torch.pi * n * f0 * t_tensor) / n
    return (4 / torch.pi) * result


# Create the time vector
# np.linspace generates evenly spaced numbers over a specified interval.
# We use endpoint=False because the interval is periodic.
t = np.linspace(0.0, T, N, endpoint=False)

# Generate the original square wave
# 先将 numpy 数组 t 转换为 PyTorch 张量
t_tensor = torch.tensor(t, dtype=torch.float64, device=device)

# 调用时传入 t_tensor 和 f0，并将结果转回 numpy 数组供下方画图使用
square = square_wave_pytorch(t_tensor, f0).cpu().numpy()

plt.figure(figsize=(12, 8))
# Plot the original square wave
plt.subplot(2, 3, 1)
plt.plot(t, square, 'k', label="Square wave")
plt.title("Original Square Wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()
# Plot Fourier reconstructions under different number of harmonics
for i, Nh in enumerate(harmonics, start=2):
    plt.subplot(2, 3, i)
    y = square_wave_fourier_pytorch(t_tensor, f0, Nh).cpu().numpy()
    plt.plot(t, y, label=f"N={Nh} harmonics")
    plt.plot(t, square, 'k--', alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation with N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

# 2. Apply the DFT and time the execution
def naive_dft_pytorch_gpu(x_tensor):
    """
    使用 PyTorch 张量操作在 GPU 上计算 DFT。
    通过矩阵运算替代显式的 for 循环以发挥 GPU 的并行加速能力。
    """
    N = x_tensor.shape[0]

    # 生成 0 到 N-1 的序列，并确保它们在指定的 GPU 设备上
    n = torch.arange(N, device=x_tensor.device)
    k = torch.arange(N, device=x_tensor.device).view(-1, 1)  # 变成列向量，用于广播

    # 核心 DFT 公式张量化：计算所有 k 和 n 组合的角度矩阵
    angle = -2j * torch.pi * k * n / N

    # 指数运算并与输入信号进行矩阵乘法
    # 指数运算，并强制将 M 的精度提升为 complex128，与输入信号对齐
    M = torch.exp(angle).to(torch.complex128)
    X = torch.mv(M, x_tensor.to(torch.complex128))

    return X
# Construct a square wave using 50 harmonics
signal = square_wave_fourier_pytorch(t_tensor, f0, 50).cpu().numpy()
# Time the naive DFT implementation
start_time_naive = time.time()
dft_result = naive_dft_pytorch_gpu(torch.tensor(signal, device=device)).cpu().numpy()
end_time_naive = time.time()
naive_duration = end_time_naive - start_time_naive
# Time NumPy's FFT implementation
start_time_fft = time.time()
fft_result = np.fft.fft(signal)
end_time_fft = time.time()
fft_duration = end_time_fft - start_time_fft
# 3. Print Timings and Verification
print("---DFT/FFT Performance Comparison---")
print(f"Naive DFT Execution Time: {naive_duration:.6f} seconds")
print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
# It's possible for the FFT to be so fast that the duration is 0.0, so we handle that case.
if fft_duration > 0:
    print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
else:
    print("FFT was too fast to measure a significant duration difference.")
# Check if our implementation is close to NumPy's result
# np.allclose is used for comparing floating-point arrays.
print(f"\nOur DFT implementation is close to NumPy's FFT: {np.allclose(dft_result, fft_result)}")
# 4. Prepare for Plotting
# Generate the frequency axis for the plot.
# np.fft.fftfreq returns the DFT sample frequencies.
# We only need the first half of the frequencies (the positive ones) due to symmetry.
xf = np.fft.fftfreq(N, d=T / N)[:N // 2]
# We normalize the magnitude by N and multiply by 2 to get the correct amplitude.
magnitude = 2.0 / N * np.abs(dft_result[0:N // 2])
# 5. Visualize the Results
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
# Plot the original time-domain signal
ax1.plot(t, signal, color='c')
ax1.set_title('Input Sine Wave Signal', fontsize=16)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim(0, 1.0)  # Show a few cycles of the sine wave
ax1.grid(True)
# Plot the frequency-domain signal (magnitude of the DFT)
ax2.stem(xf, magnitude, basefmt=" ")
ax2.set_title(
'Discrete Fourier Transform (Magnitude Spectrum)',
fontsize=16
)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Magnitude', fontsize=12)
ax2.set_xlim(0, 50) # Focus on lower frequencies
ax2.grid(True)
# Add vertical lines for the first ten frequencies
for i in range(20):
    if i < len(xf) and i % 2 == 1: # Only plot odd harmonics
        ax2.axvline(
            xf[i], color='r', linestyle='--', alpha=0.7,
            label=f'f{i}: {i}* f0 = {xf[i]:.1f} Hz'
            )
# Only show labels for first 3 frequencies to avoid cluttering
ax2.legend()
plt.tight_layout()
plt.show()

# 1. 准备 PyTorch 格式的数据并移动到 GPU
# 将原来 numpy 生成的时间 t 转换为 tensor
t_tensor = torch.tensor(t, dtype=torch.float64, device=device)
f0 = 1

# 生成包含 50 个谐波的方波信号 (在 GPU 上)
signal_tensor = square_wave_fourier_pytorch(t_tensor, f0, 50)

# 2. 计时测试：PyTorch GPU Naive DFT
# 注意：在 GPU 计时前最好进行一次预热 (warm-up)，确保 GPU 已经初始化
_ = naive_dft_pytorch_gpu(signal_tensor)

start_time_gpu = time.time()
dft_result_gpu = naive_dft_pytorch_gpu(signal_tensor)
# torch.cuda.synchronize() # 如果在真实 CUDA 环境中，需取消注释以确保准确计时
end_time_gpu = time.time()
gpu_duration = end_time_gpu - start_time_gpu

print("\n--- 增加 PyTorch GPU 的性能对比 ---")
print(f"1. Naive DFT (NumPy/CPU) Time: {naive_duration:.6f} seconds")
print(f"2. Naive DFT (PyTorch/GPU) Time: {gpu_duration:.6f} seconds")
print(f"3. Built-in FFT (NumPy/CPU) Time: {fft_duration:.6f} seconds")

# 验证 GPU 结果是否与 NumPy 内置 FFT 结果一致 (将 GPU tensor 转回 cpu 的 numpy 数组)
print(f"PyTorch GPU DFT is close to NumPy's FFT: {np.allclose(dft_result_gpu.cpu().numpy(), fft_result)}")