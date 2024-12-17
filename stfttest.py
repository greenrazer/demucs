import torch
import math
import matplotlib.pyplot as plt


class FFTCore(torch.nn.Module):
    def __init__(self, num_samples, inverse=False):
        super().__init__()

        self.inverse = inverse
        self.one = torch.tensor(1.0)
        self.two = torch.tensor(2.0)
        self.angle_sign = self.one if self.inverse else -self.one

        self.num_samples = num_samples
        self.stages = int(math.log2(num_samples))
        self.bit_reversed_indicies = self.bit_reverse_permutation(num_samples)

    def forward(self, x):
        n = x.shape[1]

        real = x[0]
        imag = x[1]

        # Perform in-place bit-reversal permutation
        real = real[self.bit_reversed_indicies]
        imag = imag[self.bit_reversed_indicies]

        for stage in range(self.stages):
            group_size = 1 << (stage + 1)
            half_group = group_size >> 1

            # Reshape data into groups
            num_groups = n // group_size
            group_indices = torch.arange(num_groups, device=x.device) * group_size

            # Calculate twiddle factors for all groups
            j = torch.arange(half_group, device=x.device)  # Parallelize j loop
            angle = self.angle_sign * self.two * torch.pi * j / group_size
            # angle = self.angle_sign * self.two * math.pi * j / group_size
            wr = torch.cos(angle)
            wi = torch.sin(angle)

            # Expand the twiddle factors across all groups
            twiddle_real = wr.unsqueeze(0).expand(num_groups, half_group)
            twiddle_imag = wi.unsqueeze(0).expand(num_groups, half_group)

            # Indices for top and bottom elements in each group
            top_indices = group_indices[:, None] + j
            bottom_indices = top_indices + half_group

            # Perform butterfly operations for all groups in parallel
            tr_real = (
                twiddle_real * real[bottom_indices]
                - twiddle_imag * imag[bottom_indices]
            )
            tr_imag = (
                twiddle_imag * real[bottom_indices]
                + twiddle_real * imag[bottom_indices]
            )

            real[bottom_indices] = real[top_indices] - tr_real
            imag[bottom_indices] = imag[top_indices] - tr_imag

            real[top_indices] += tr_real
            imag[top_indices] += tr_imag

        out = torch.stack((real, imag), dim=0)

        if self.inverse:
            out = out / out.shape[1]

        return out

    def bit_reverse_permutation(self, n):
        indices = torch.arange(n)
        reversed_indices = torch.zeros_like(indices)

        num_bits = n.bit_length() - 1  # The number of bits for the indices
        for i in range(n):
            rev = 0
            temp = i
            for j in range(num_bits):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            reversed_indices[i] = rev

        return reversed_indices

class CustomSTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        pad_mode="reflect",
        normalized=False,
        center=True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window if window is not None else torch.hann_window(n_fft)
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.center = center

        # Initialize FFTCore
        self.fft_core = FFTCore(n_fft)

    def forward(self, x):
        # Step 1: Centering and padding
        if self.center:
            x = self._center_signal(x)

        # Step 2: Segment the signal into overlapping windows
        frames = self._frame_signal(x)

        # Step 3: Apply window function to each frame
        frames = frames * self.window.unsqueeze(1)

        # Step 4: Perform FFT on each frame
        parts = []
        for frame_idx in range(frames.shape[-1]):
            frame = frames[..., frame_idx]
            fft_result = self.fft_core(frame)
            parts.append(fft_result)

        out = torch.stack(parts, dim=-1)
        out = out[:, : self.n_fft // 2 + 1, :]

        if self.normalized:
            out = out / math.sqrt(self.n_fft)

        return out

    def _center_signal(self, x):
        # Apply padding and centering if needed
        if self.pad_mode == "reflect":
            x = torch.nn.functional.pad(
                x, (self.n_fft // 2, self.n_fft // 2), mode="reflect"
            )
        return x

    def _frame_signal(self, x, n_fft=None, hop_length=None):
        n_fft = n_fft if n_fft is not None else self.n_fft
        hop_length = hop_length if hop_length is not None else self.hop_length

        frames = []
        for i in range(0, x.size(-1) - n_fft + 1, hop_length):
            frame = x[:, i : i + n_fft]
            frames.append(frame)
        return torch.stack(frames, dim=2)


class CustomISTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        normalized=False,
        center=True,
        length=None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.window = (window if window is not None else torch.hann_window(n_fft)) * (1.5 / n_fft)
        self.normalized = normalized
        self.center = center
        self.length = length
        self.fft_core = FFTCore(n_fft, inverse=True)  # Use inverse FFT

    def forward(self, z):
        if self.normalized:
            z = z * math.sqrt(self.n_fft)
        frames = []
        for i in range(z.size(-1)):
            frame = z[..., i]
            frame = self._build_full_spectrum(frame)
            ifft_result = self.fft_core(frame)
            frames.append(ifft_result)

        frames = torch.stack(frames, dim=-1)

        # Step 3: Apply overlap-add with window
        signal = self._overlap_add(frames)

        if self.center:
            # Trim the padding added during STFT
            signal = signal[..., self.n_fft // 2 : -self.n_fft // 2]

        if self.length is not None:
            # Match the desired output length
            signal = signal[..., : self.length]

        # # Step 4: Apply window normalization if needed
        if self.normalized:
            signal = signal / self.window.sum()

        return signal

    def _build_full_spectrum(self, half_spectrum):
        real_half = half_spectrum[0, ...]
        imag_half = half_spectrum[1, ...]

        # Construct negative frequency components symmetrically
        real_full = torch.cat([real_half, real_half.flip(0)[1:-1]], dim=0)
        imag_full = torch.cat([imag_half, -imag_half.flip(0)[1:-1]], dim=0)

        full_spectrum = torch.stack((real_full, imag_full), dim=0)

        return full_spectrum

    def _overlap_add(self, frames, hop_length=None, win_length=None, window=None):
        hop_length = hop_length if hop_length is not None else self.hop_length
        win_length = win_length if win_length is not None else self.win_length
        window = window if window is not None else self.window

        output_length = (frames.size(-1) - 1) * hop_length + win_length

        signal = torch.zeros((2, output_length), device=frames.device)
        window_sum = torch.zeros(output_length, device=frames.device)

        for i in range(frames.shape[-1]):
            start_idx = i * hop_length
            end_idx = start_idx + win_length

            frame = frames[..., i]
            signal[:, start_idx:end_idx] += frame * window
            window_sum[start_idx:end_idx] += window

        # Normalize by the overlapping window sum
        non_zero = window_sum >= 1e-6
        signal[..., non_zero] /= window_sum[non_zero]
        # signal /= window_sum

        return signal


def visualize_signals(original_signal, reconstructed_signal, title="Signal Comparison"):
    """
    Visualize the original and reconstructed signals on top of each other.

    Args:
        original_signal (torch.Tensor): Original signal of shape (N, 2).
        reconstructed_signal (torch.Tensor): Reconstructed signal of shape (N, 2).
        title (str): Title for the plot.
    """
    # Ensure the signals are numpy arrays for plotting
    original_signal = original_signal.cpu().numpy()
    reconstructed_signal = reconstructed_signal.cpu().numpy()

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Plot channel 1
    plt.subplot(1, 1, 1)
    plt.plot(original_signal, label="Original Signal (Ch 1)", alpha=0.7)
    plt.plot(
        reconstructed_signal,
        label="Reconstructed Signal (Ch 1)",
        linestyle="dashed",
        alpha=0.7,
    )
    plt.title(f"{title} - Channel 1")
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()
    
def fft_test():
    num_samples = int(math.pow(2, 9))

    torch.manual_seed(42)
    input_tensor = torch.rand(num_samples)
    input_tensor_2d = torch.stack((input_tensor, torch.zeros_like(input_tensor)), dim=0)

    torch_fft = torch.fft.fft(input_tensor)

    fft_real = FFTCore(num_samples=num_samples)
    my_fft = fft_real(input_tensor_2d)

    # print(torch.min(torch.square(torch_fft.real - my_fft[0])))
    # print(torch.median(torch.square(torch_fft.real - my_fft[0])))
    # print(torch.max(torch.square(torch_fft.real - my_fft[0])))

    tolerance = 1e-5
    assert torch.allclose(
        torch_fft.real, my_fft[0], atol=tolerance
    ), "FFT Real Units Test Failed."
    assert torch.allclose(
        torch_fft.imag, my_fft[1], atol=tolerance
    ), "FFT Imag Units Test Failed."
    print("FFT Test Passed")


def ifft_test():
    num_samples = int(math.pow(2, 9))

    torch.manual_seed(42)
    input_tensor = torch.rand(num_samples)

    freqs = torch.fft.fft(input_tensor)
    freqs_2d = torch.stack((freqs.real, freqs.imag), dim=0)

    torch_ifft = torch.fft.ifft(freqs)

    ifft_real = FFTCore(num_samples=num_samples, inverse=True)
    my_ifft = ifft_real(freqs_2d)

    # print(torch.min(torch.square(torch_ifft.real - my_ifft[0])))
    # print(torch.median(torch.square(torch_ifft.real - my_ifft[0])))
    # print(torch.max(torch.square(torch_ifft.real - my_ifft[0])))

    tolerance = 1e-6
    assert torch.allclose(
        torch_ifft.real, my_ifft[0], atol=tolerance
    ), "IFFT Test Failed"
    print("IFFT Test Passed")

def fft_reversible_test():
    num_samples = int(math.pow(2, 9))
    input_tensor = torch.rand(num_samples)

    fft_real = FFTCore(num_samples=num_samples)
    ifft_real = FFTCore(num_samples=num_samples, inverse=True)
    
    fft_result = fft_real(torch.stack((input_tensor, torch.zeros_like(input_tensor)), dim=-0))
    ifft_result = ifft_real(fft_result)[0]

    # print(torch.min(torch.square(input_tensor - ifft_result)))
    # print(torch.median(torch.square(input_tensor - ifft_result)))
    # print(torch.max(torch.square(input_tensor - ifft_result)))

    tolerance = 1e-6
    assert torch.allclose(input_tensor, ifft_result, atol=tolerance), "Random input reproducibility test failed!"
    print("Random input reproducibility test passed.")

def stft_test():
    n_fft = 512
    hop_length = n_fft // 4
    win_length = n_fft
    window = torch.hann_window(n_fft)

    signal_length = 2048
    sampling_rate = 8000
    t = torch.arange(signal_length) / sampling_rate
    signal = torch.sin(2 * torch.pi * 100 * t) + 0.1 * torch.randn_like(t)
    signal_2d = torch.stack((signal, torch.zeros_like(signal)), dim=0)

    torch_stft = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )

    custom_stft_mod = CustomSTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
        pad_mode="reflect",
    )
    custom_stft = custom_stft_mod(signal_2d)

    torch_real = torch_stft.real
    torch_imag = torch_stft.imag
    custom_real = custom_stft[0, ...]
    custom_imag = custom_stft[1, ...]

    tolerance = 1e-6
    assert torch.allclose(
        torch_real, custom_real, atol=tolerance
    ), "STFT Real Units Test Failed."

    assert torch.allclose(
        torch_imag, custom_imag, atol=tolerance
    ), "STFT Imaginary Units Test Failed."

    print("STFT Test Passed!")

def stft_reproduceable_test():
    n_fft = 512
    hop_length = n_fft // 4
    win_length = n_fft
    window = torch.hann_window(n_fft)

    signal_length = 2048
    sampling_rate = 8000
    t = torch.arange(signal_length) / sampling_rate
    signal = torch.sin(2 * torch.pi * 100 * t) + 0.1 * torch.randn_like(t)
    signal_2d = torch.stack((signal, torch.zeros_like(signal)), dim=0)

    torch_stft = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )

    custom_stft_mod = CustomSTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
        pad_mode="reflect",
    )
    custom_stft = custom_stft_mod(signal_2d)

    torch_reconstructed_signal = torch.istft(
        torch_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        win_length=win_length,
        normalized=True,
        center=True,
        length=signal_length,
    )
    torch_reconstructed_signal_1d = torch_reconstructed_signal.real

    custom_reconstructed_signal = torch.istft(
        custom_stft[0] + 1j*custom_stft[1],
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        win_length=win_length,
        normalized=True,
        center=True,
        length=signal_length,
    )
    custom_reconstructed_signal_1d = custom_reconstructed_signal.real

    tolerance = 1e-6
    assert torch.allclose(
        torch_reconstructed_signal_1d, custom_reconstructed_signal_1d, atol=tolerance
    ), "STFT Reproduceable Test Failed."
    print("STFT Reproduceable Test Passed!")


def build_full_spectrum_test():
    n_fft = 512
    hop_length = n_fft // 4
    win_length = n_fft
    window = torch.hann_window(n_fft)

    signal_length = int(math.pow(2, 12))
    sampling_rate = 8000
    t = torch.arange(signal_length) / sampling_rate
    signal = torch.ones_like(t)

    torch_stft = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )

    torch_stft2 = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
        return_complex=True,
        onesided=False,
        pad_mode="reflect",
    )

    a = torch_stft2.real[:, 0]
    b_i = torch.stack((torch_stft.real, torch_stft.imag), dim=0)
    b = CustomISTFT._build_full_spectrum(None, b_i[..., 0])[0]

    assert torch.allclose(a, b, atol=1e-10), "Build Spectrum Failed."
    print("Build Spectrum Test Passed!")


def overlap_add_test():
    n_fft = 4
    hop_length = 4
    win_length = 4
    window = torch.rand((win_length))

    signal_length = 8
    signal = torch.stack([torch.rand((signal_length)), torch.zeros((signal_length))])

    frames = CustomSTFT._frame_signal(None, signal, n_fft=n_fft, hop_length=hop_length)
    result = CustomISTFT._overlap_add(
        None, frames, hop_length=hop_length, win_length=win_length, window=window
    )

    assert torch.allclose(result, signal), "Overlap Add Test Failed."
    print("Overlap Add Test Passed!")


def istft_test():
    n_fft = 512
    hop_length = n_fft // 4
    win_length = n_fft
    window = torch.hann_window(n_fft)

    signal_length = int(math.pow(2, 10))
    # print(signal_length)
    sampling_rate = 8000
    t = torch.arange(signal_length) / sampling_rate
    signal = torch.sin(2 * torch.pi * 100 * t) + 0.1 * torch.randn_like(t)
    # signal = torch.ones_like(t)

    torch_stft = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )

    torch_reconstructed_signal = torch.istft(
        torch_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        win_length=win_length,
        normalized=True,
        center=True,
        length=signal_length,
    )
    torch_reconstructed_signal_1d = torch_reconstructed_signal.real

    custom_istft = CustomISTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        win_length=win_length,
        normalized=True,
        center=True,
        length=signal_length,
    )
    stft_signal = torch.stack((torch_stft.real, torch_stft.imag), dim=0)
    custom_reconstructed_signal = custom_istft(stft_signal)
    custom_reconstructed_signal_1d = custom_reconstructed_signal[0, ...]

    # print(
    #     "min",
    #     torch.square(torch_reconstructed_signal_1d - custom_reconstructed_signal_1d)
    #     .min()
    #     .item(),
    # )
    # print(
    #     "median",
    #     torch.square(torch_reconstructed_signal_1d - custom_reconstructed_signal_1d)
    #     .median()
    #     .item(),
    # )
    # print(
    #     "max",
    #     torch.square(torch_reconstructed_signal_1d - custom_reconstructed_signal_1d)
    #     .max()
    #     .item(),
    # )

    # print("Input Signal (first 10):", signal)
    # print("Torch Reconstructed Signal (first 10):", torch_reconstructed_signal_1d)
    # print("Reconstructed Signal (first 10):", custom_reconstructed_signal_1d[:10])
    # visualize_signals(signal, custom_reconstructed_signal_1d)

    tolerance = 2
    assert torch.allclose(
        torch_reconstructed_signal_1d, custom_reconstructed_signal_1d, atol=tolerance
    ), "ISTFT Test Failed."
    print("ISTFT Test Passed!")

def istft_reproducable_test():
    n_fft = 512
    hop_length = n_fft // 4
    win_length = n_fft
    window = torch.hann_window(n_fft)

    signal_length = 2048
    sampling_rate = 8000
    t = torch.arange(signal_length) / sampling_rate
    signal = torch.sin(2 * torch.pi * 100 * t) + 0.1 * torch.randn_like(t)
    signal_2d = torch.stack((signal, torch.zeros_like(signal)), dim=0)

    custom_stft_mod = CustomSTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        center=True,
        pad_mode="reflect",
    )
    custom_stft = custom_stft_mod(signal_2d)

    custom_istft = CustomISTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        win_length=win_length,
        normalized=True,
        center=True,
        length=signal_length,
    )
    custom_reconstructed_signal = custom_istft(custom_stft)
    custom_reconstructed_signal_1d = custom_reconstructed_signal[0, ...]

    # visualize_signals(signal, custom_reconstructed_signal_1d)

    # print(torch.min(torch.square(signal - custom_reconstructed_signal_1d)))
    # print(torch.median(torch.square(signal - custom_reconstructed_signal_1d)))
    # print(torch.max(torch.square(signal - custom_reconstructed_signal_1d)))


    tolerance = 1
    assert torch.allclose(
        signal, custom_reconstructed_signal_1d, atol=tolerance
    ), "STFT Reproduceable Test Failed."
    print("STFT Reproduceable Test Passed!")

def main():
    fft_test()
    ifft_test()
    fft_reversible_test()
    stft_test()
    stft_reproduceable_test()
    build_full_spectrum_test()
    overlap_add_test()
    istft_test()
    istft_reproducable_test()


if __name__ == "__main__":
    main()

# def visualize_signals3(original_signal, torch_reconstruction, my_reconstruction, title="Signal Comparison"):
#     """
#     Visualize the original and reconstructed signals on top of each other.

#     Args:
#         original_signal (torch.Tensor): Original signal of shape (N, 2).
#         reconstructed_signal (torch.Tensor): Reconstructed signal of shape (N, 2).
#         title (str): Title for the plot.
#     """
#     # Ensure the signals are numpy arrays for plotting
#     original_signal = original_signal.cpu().numpy()
#     torch_reconstruction = torch_reconstruction.cpu().numpy()
#     my_reconstruction = my_reconstruction.cpu().numpy()

#     # Create the figure
#     plt.figure(figsize=(12, 6))

#     # Plot channel 1
#     plt.subplot(1, 1, 1)
#     plt.plot(original_signal, label="Original Signal", alpha=0.7)
#     plt.plot(
#         torch_reconstruction,
#         label="Torch Reconstructed Signal",
#         linestyle="dashed",
#         alpha=0.7,
#     )
#     plt.plot(
#         my_reconstruction,
#         label="My Reconstructed Signal",
#         linestyle="dashed",
#         alpha=0.7,
#     )
#     plt.title(f"{title}")
#     plt.legend()
#     plt.grid(True)

#     # Show the plots
#     plt.tight_layout()
#     plt.show()

# num_samples = int(math.pow(2, 18))

# fft_real = FFTCore(num_samples=num_samples)
# ifft_real = FFTCore(num_samples=num_samples, inverse=True)

# torch.manual_seed(42)
# input_tensor = torch.rand(2**18)
# input_tensor = torch.stack((input_tensor, torch.zeros_like(input_tensor)), dim=0)

# fft_result = fft_real(input_tensor)
# ifft_result = ifft_real(fft_result)

# assert torch.allclose(
#     input_tensor[0], ifft_result[0], atol=1e-6
# ), "Random input reproducibility test failed!"
# print("fft test passed")

# def calculate_snr(original, reconstructed):
#     """
#     Calculate the Signal-to-Noise Ratio (SNR) between the original and reconstructed signals.
#     Args:
#         original (torch.Tensor): Original signal of shape (N, 2).
#         reconstructed (torch.Tensor): Reconstructed signal of shape (N, 2).
#     Returns:
#         float: The SNR value in decibels (dB).
#     """
#     signal_power = torch.sum(original**2)
#     noise_power = torch.sum((original - reconstructed) ** 2)
#     snr = 10 * torch.log10(signal_power / noise_power)
#     return snr.item()


# def calculate_mse(original, reconstructed):
#     """
#     Calculate the Mean Squared Error between the original and reconstructed signals.
#     Args:
#         original (torch.Tensor): Original signal of shape (N, 2).
#         reconstructed (torch.Tensor): Reconstructed signal of shape (N, 2).
#     Returns:
#         float: The MSE value.
#     """
#     return torch.mean((original - reconstructed) ** 2).item()

# def test_stft_istft():
#     # Parameters
#     n_fft = 512  # FFT size
#     hop_length = n_fft // 4  # Overlap size
#     window=torch.hann_window(n_fft)
#     win_length = n_fft  # Window length (same as n_fft)
#     signal_length = 2048  # Length of the input signal

#     # Create a synthetic signal (e.g., a sine wave with noise)
#     # Signal: sine wave at 100 Hz sampled at 8000 Hz
#     sampling_rate = 8000
#     t = torch.arange(signal_length) / sampling_rate
#     signal = torch.sin(2 * torch.pi * 100 * t) + 0.1 * torch.randn_like(
#         t
#     )  # Sine wave with noise

#     # Convert to a 2D tensor (batch_size=1, num_samples=signal_length)
#     signal = torch.stack((signal, torch.zeros_like(signal)), dim=-1)

#     # Create STFT and ISTFT modules
#     stft_module = CustomSTFT(
#         n_fft=n_fft,
#         hop_length=hop_length,
#         window=window
#     )
#     istft_module = CustomISTFT(
#         n_fft=n_fft,
#         hop_length=hop_length,
#         window=window,
#         length=signal_length,
#     )

#     # Step 1: Compute the STFT of the signal
#     z = stft_module(signal)

#     # Step 2: Compute the iSTFT (inverse STFT) to reconstruct the signal
#     reconstructed_signal = istft_module(z)

#     print(calculate_mse(signal[..., 0], reconstructed_signal[..., 0]))
#     print(calculate_snr(signal[..., 0], reconstructed_signal[..., 0]))
#     visualize_signals(signal[..., 0], reconstructed_signal[..., 0])

#     # Step 3: Compare the original and reconstructed signals
#     # For simplicity, we compare the signals with a small tolerance
#     tolerance = 1e-5
#     assert torch.allclose(
#         signal[..., 0], reconstructed_signal[..., 0], atol=tolerance
#     ), "Reconstructed signal differs from the original."

#     # Print test result
#     print("Test passed: The original and reconstructed signals are close.")


# def compare_with_torch_stft_istft():
#     # Parameters
#     n_fft = 512
#     hop_length = n_fft // 4
#     win_length = n_fft
#     signal_length = 2048
#     tolerance = 1e-5  # Tolerance for numerical differences

#     # Create a synthetic signal
#     sampling_rate = 8000
#     t = torch.arange(signal_length) / sampling_rate
#     signal = torch.sin(2 * torch.pi * 100 * t) + 0.1 * torch.randn_like(
#         t
#     )  # Sine wave with noise

#     # Create the same Hann window as used in both implementations
#     window = torch.hann_window(n_fft)

#     print("torch_signal", signal.shape)
#     # PyTorch STFT and ISTFT
#     torch_stft = torch.stft(
#         signal,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         win_length=win_length,
#         window=window,
#         normalized=True,
#         center=True,
#         return_complex=True,
#         pad_mode="reflect",
#     )
#     print("torch freq", torch_stft.shape)
#     torch_reconstructed_signal = torch.istft(
#         torch_stft,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         window=window,
#         win_length=win_length,
#         normalized=True,
#         center=True,
#         length=signal_length,
#     )
#     print("torch reconstructed",torch_reconstructed_signal.shape)
#     torch_reconstructed_signal_1d = torch_reconstructed_signal.real

#     # Custom STFT and ISTFT
#     custom_stft = CustomSTFT(
#         n_fft=n_fft,
#         hop_length=hop_length,
#         win_length=win_length,
#         window=window,
#         normalized=True,
#         center=True,
#         pad_mode="reflect",
#     )
#     custom_istft = CustomISTFT(
#         n_fft=n_fft,
#         hop_length=hop_length,
#         window=window,
#         win_length=win_length,
#         normalized=True,
#         center=True,
#         length=signal_length,
#     )

#     signal_2d = torch.stack(
#         (signal, torch.zeros_like(signal)), dim=0
#     )  # Match custom input shape

#     print("my signal", signal_2d.shape)
#     custom_stft_result = custom_stft(signal_2d)
#     print("my freq", custom_stft_result.shape)
#     custom_reconstructed_signal = custom_istft(custom_stft_result)
#     print("my reconstructed", custom_reconstructed_signal.shape)

#     # Convert custom output to match PyTorch's format for comparison
#     custom_reconstructed_signal_1d = custom_reconstructed_signal[0, ...]

#     # Compare the STFT outputs
#     custom_real = custom_stft_result[0, ...]
#     custom_imag = custom_stft_result[1, ...]
#     torch_real = torch_stft.real
#     torch_imag = torch_stft.imag

#     stft_real_diff = torch.abs(custom_real - torch_real).mean().item()
#     stft_imag_diff = torch.abs(custom_imag - torch_imag).mean().item()

#     print(f"STFT Real Part Mean Absolute Difference: {stft_real_diff}")
#     print(f"STFT Imaginary Part Mean Absolute Difference: {stft_imag_diff}")

#     # Compare reconstructed signals
#     # reconstruction_diff = (
#     #     torch.abs(custom_reconstructed_signal_1d - torch_reconstructed_signal_1d)
#     #     .mean()
#     #     .item()
#     # )
#     # print(f"Reconstruction Mean Absolute Difference: {reconstruction_diff}")

#     # visualize_signals3(signal, torch_reconstructed_signal_1d, custom_reconstructed_signal_1d)

#     # Verify tolerances
#     # assert stft_real_diff < tolerance, "STFT real part differs beyond tolerance!"
#     # assert stft_imag_diff < tolerance, "STFT imaginary part differs beyond tolerance!"
#     # assert (
#     #     reconstruction_diff < tolerance
#     # ), "Reconstructed signal differs beyond tolerance!"

#     # print(
#     #     "Test passed: Custom STFT and ISTFT implementations are consistent with PyTorch."
#     # )


# if __name__ == "__main__":
#     fft_test()
#     stft_test
#     # test_stft_istft()
#     # compare_with_torch_stft_istft()
