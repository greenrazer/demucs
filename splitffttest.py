import torch
import math
import matplotlib.pyplot as plt

class FFTCore(torch.nn.Module):
    def __init__(self, inverse = False):
        super().__init__()

        self.inverse = inverse
        self.one = torch.tensor(1.0)
        self.two = torch.tensor(2.0)
        self.angle_sign = self.one if self.inverse else -self.one

    def forward(self, x):
        n = x.shape[0]

        real = x[:, 0]
        imag = x[:, 1]
        
        # Perform in-place bit-reversal permutation
        indices = self.bit_reverse_permutation(n)
        real = real[indices]
        imag = imag[indices]

        stages = int(math.log2(n))
        
        for stage in range(stages):
            group_size = 1 << (stage + 1)
            half_group = group_size >> 1

            # Reshape data into groups
            num_groups = n // group_size
            group_indices = torch.arange(num_groups, device=x.device) * group_size

            # Calculate twiddle factors for all groups
            j = torch.arange(half_group, device=x.device)  # Parallelize j loop
            angle = self.angle_sign * self.two * torch.pi * j / group_size
            wr = torch.cos(angle)
            wi = torch.sin(angle)

            # Expand the twiddle factors across all groups
            twiddle_real = wr.unsqueeze(0).expand(num_groups, half_group)
            twiddle_imag = wi.unsqueeze(0).expand(num_groups, half_group)

            # Indices for top and bottom elements in each group
            top_indices = group_indices[:, None] + j
            bottom_indices = top_indices + half_group

            # Perform butterfly operations for all groups in parallel
            tr_real = twiddle_real * real[bottom_indices] - twiddle_imag * imag[bottom_indices]
            tr_imag = twiddle_imag * real[bottom_indices] + twiddle_real * imag[bottom_indices]
            
            real[bottom_indices] = real[top_indices] - tr_real
            imag[bottom_indices] = imag[top_indices] - tr_imag
            
            real[top_indices] += tr_real
            imag[top_indices] += tr_imag
        
        out = torch.stack((real, imag), dim=-1)

        if self.inverse:
            out = out / out.shape[0]

        return out
    
    def bit_reverse_permutation(self, n):
        indices = torch.arange(n)
        reversed_indices = torch.zeros_like(indices)
        
        for i in range(n):
            rev = 0
            temp = i
            for j in range(int(math.log2(n))):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            reversed_indices[i] = rev
        
        return reversed_indices


class Wrap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = FFTCore()
        self.inv_fft = FFTCore(inverse=True)

    def forward(self, x):
        out = self.fft(x)
        return self.inv_fft(out)
    
# if __name__ == "__main__":
#     torch.set_grad_enabled(False)
#     model = Wrap()
#     a = torch.rand(512)
#     inp = torch.stack((a, torch.zeros_like(a)), dim=-1)
#     out = model(inp)
#     b = out[..., 0]
#     print(a-b)
#     print("---")

#     tm = torch.jit.trace(model, (inp,))

#     tm.save("model.pt")

fft_real = FFTCore()
ifft_real = FFTCore(inverse=True)
def comprehensive_fft_tests():
    """
    Comprehensive test suite for FFT implementation
    """
    print("\n--- Comprehensive FFT Tests ---")
    
    # Utility function for detailed comparison
    def detailed_comparison(original, reconstructed, test_name):
        """
        Provide detailed comparison of original and reconstructed signals
        """
        print(f"\nDetailed Comparison for {test_name}")
        
        # Compute absolute differences
        abs_diff = torch.abs(original - reconstructed)
        
        # Basic statistics
        print("Absolute Difference Statistics:")
        print(f"Max Difference: {torch.max(abs_diff)}")
        print(f"Mean Difference: {torch.mean(abs_diff)}")
        print(f"Standard Deviation of Difference: {torch.std(abs_diff)}")
        
        # Plot original and reconstructed signals
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title(f"{test_name} - Original Signal")
        plt.plot(original.numpy(), label='Original')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.title(f"{test_name} - Reconstructed Signal")
        plt.plot(reconstructed.numpy(), label='Reconstructed')
        plt.plot(original.numpy(), label='Original', linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Detailed index-wise comparison
        print("\nIndex-wise Differences:")
        for i, (orig, recon, diff) in enumerate(zip(original, reconstructed, abs_diff)):
            if diff > 1e-6:
                print(f"Index {i}: Original={orig:.6f}, Reconstructed={recon:.6f}, Difference={diff:.6f}")
    
    # Test 1: Zero input
    def test_zero_input():
        print("Test 1: Zero Input")
        input_tensor = torch.zeros(512)
        fft_result = fft_real(torch.stack((input_tensor, torch.zeros_like(input_tensor)), dim=-1))
        ifft_result = ifft_real(fft_result)[..., 0]
        
        assert torch.allclose(input_tensor, ifft_result, atol=1e-6), "Zero input test failed!"
        print("Zero input test passed.")
    
    # Test 2: Constant input
    def test_constant_input():
        print("Test 2: Constant Input")
        input_tensor = torch.ones(512)
        fft_result = fft_real(torch.stack((input_tensor, torch.zeros_like(input_tensor)), dim=-1))
        ifft_result = ifft_real(fft_result)[..., 0]

        detailed_comparison(input_tensor, ifft_result, "Const")
        
        assert torch.allclose(input_tensor, ifft_result, atol=1e-6), "Constant input test failed!"
        print("Constant input test passed.")
    
    # Test 3: Sine Wave with Multiple Frequencies
    def test_multi_frequency_sine():
        print("Test 3: Multi-Frequency Sine Wave")
        n = 512
        t = torch.arange(n, dtype=torch.float32)
        
        # Composite sine wave with multiple frequencies
        sine_wave = (
            torch.sin(2 * torch.pi * 2 * t / n) +  # 2 Hz
            0.5 * torch.sin(2 * torch.pi * 3 * t / n) +  # 3 Hz with half amplitude
            0.25 * torch.sin(2 * torch.pi * 5 * t / n)   # 5 Hz with quarter amplitude
        )
        
        fft_result = fft_real(torch.stack((sine_wave, torch.zeros_like(sine_wave)), dim=-1))
        
        # Visualize frequency spectrum
        magnitudes = torch.norm(fft_result, p=2, dim=1)
        print("Magnitude Spectrum:", magnitudes)
        
        ifft_result = ifft_real(fft_result)[..., 0]
        
        # Detailed comparison
        detailed_comparison(sine_wave, ifft_result, "Multi-Frequency Sine Wave")

        print(ifft_result)
        
        assert torch.allclose(sine_wave, ifft_result, atol=1e-6), "Multi-frequency sine wave test failed!"
        print("Multi-frequency sine wave test passed.")
    
    # Test 4: Random Input Reproducibility
    def test_random_input_reproducibility():
        print("Test 4: Random Input Reproducibility")
        torch.manual_seed(42)  # Set seed for reproducibility
        input_tensor = torch.rand(512)
        
        fft_result = fft_real(torch.stack((input_tensor, torch.zeros_like(input_tensor)), dim=-1))
        ifft_result = ifft_real(fft_result)[..., 0]
        
        # Detailed comparison
        detailed_comparison(input_tensor, ifft_result, "Random Input")
        
        assert torch.allclose(input_tensor, ifft_result, atol=1e-6), "Random input reproducibility test failed!"
        print("Random input reproducibility test passed.")
    
    # Run tests
    test_zero_input()
    test_constant_input()
    test_multi_frequency_sine()
    test_random_input_reproducibility()
    
    print("\n--- All Comprehensive FFT Tests Passed! ---")


if __name__ == "__main__":
    comprehensive_fft_tests()