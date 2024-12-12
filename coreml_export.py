import demucs.api
from demucs.api import AudioFile

import copy
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union
import torch
from torch.nn import functional as F
import subprocess
import torchaudio
import julius
import lameenc


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError(
            "The audio file has less channels than requested but is not mono."
        )
    return wav


def pad_symmetrically(tensor, target_length, offset=0, length=None):
    total_length = tensor.shape[-1]

    if length is None:
        length = total_length - offset
    else:
        length = min(total_length - offset, length)

    delta = target_length - length
    assert delta >= 0

    start = offset - delta // 2
    end = start + target_length

    correct_start = max(0, start)
    correct_end = min(total_length, end)

    pad_left = correct_start - start
    pad_right = end - correct_end

    out = F.pad(tensor[..., correct_start:correct_end], (pad_left, pad_right))
    return out.to(tensor.device)


def center_trim(tensor: torch.Tensor, reference: Union[torch.Tensor, int]):
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor


def load_audio(model, track: Path):
    sample_rate = model.samplerate
    audio_channels = model.audio_channels

    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0, samplerate=sample_rate, channels=audio_channels
        )
    except FileNotFoundError:
        errors["ffmpeg"] = "FFmpeg is not installed."
    except subprocess.CalledProcessError:
        errors["ffmpeg"] = "FFmpeg could not read the file."

    if wav is None:
        try:
            wav, sr = torchaudio.load(str(track))
        except RuntimeError as err:
            errors["torchaudio"] = err.args[0]
        else:
            wav = convert_audio_channels(wav, audio_channels)
            wav = julius.resample_frac(wav, sr, sample_rate)

    if wav is None:
        raise Exception(
            "\n".join(
                "When trying to load using {}, got the following error: {}".format(
                    backend, error
                )
                for backend, error in errors.items()
            )
        )
    return wav


def prevent_clip(wav, mode="rescale"):
    """
    different strategies for avoiding raw clipping.
    """
    if mode is None or mode == "none":
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == "rescale":
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == "clamp":
        wav = wav.clamp(-0.99, 0.99)
    elif mode == "tanh":
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def i16_pcm(wav):
    """Convert audio to 16 bits integer PCM format."""
    if wav.dtype.is_floating_point:
        return (wav.clamp_(-1, 1) * (2**15 - 1)).short()
    else:
        return wav


def encode_mp3(wav, path, samplerate=44100, bitrate=320, quality=2, verbose=False):
    """Save given audio as mp3. This should work on all OSes."""
    C, T = wav.shape
    wav = i16_pcm(wav)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(C)
    encoder.set_quality(quality)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    wav = wav.data.cpu()
    wav = wav.transpose(0, 1).numpy()
    mp3_data = encoder.encode(wav.tobytes())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


def save_audio(
    wav: torch.Tensor,
    path: Union[str, Path],
    samplerate: int,
    bitrate: int = 320,
    clip: Literal["rescale", "clamp", "tanh", "none"] = "rescale",
    bits_per_sample: Literal[16, 24, 32] = 16,
    as_float: bool = False,
    preset: Literal[2, 3, 4, 5, 6, 7] = 2,
):
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        encode_mp3(wav, path, samplerate, bitrate, preset, verbose=True)
    elif suffix == ".wav":
        if as_float:
            bits_per_sample = 32
            encoding = "PCM_F"
        else:
            encoding = "PCM_S"
        torchaudio.save(
            str(path),
            wav,
            sample_rate=samplerate,
            encoding=encoding,
            bits_per_sample=bits_per_sample,
        )
    elif suffix == ".flac":
        torchaudio.save(
            str(path), wav, sample_rate=samplerate, bits_per_sample=bits_per_sample
        )
    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")


def preprocess(
    wav: torch.Tensor,
    input_sample_rate: int,
    model_sample_rate: int,
    audio_channels: int,
):
    if input_sample_rate is not None and input_sample_rate != model_sample_rate:
        wav = convert_audio_channels(wav, audio_channels)
        wav = julius.resample_frac(wav, input_sample_rate, model_sample_rate)

    # normalize
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8

    mix = wav.unsqueeze(0)

    return (ref, mix)

class Shifts:
    def __init__(self, length, max_shift, num_shifts):
        self.num_shifts = num_shifts
        self.max_shift = max_shift
        self.length = length
        self.current_shift = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_shift < self.num_shifts:
            shift_offset = self.max_shift // (self.current_shift + 1)
            shift_length = self.length + self.max_shift - shift_offset
            self.current_shift += 1
            return shift_offset, shift_length
        else:
            raise StopIteration

class Chunks:
    def __init__(self, length, max_shift, shift_offset, shift_length, chunk_overlap, chunk_segment_length):
        self.length = length
        self.shift_length = shift_length
        self.chunk_overlap = chunk_overlap
        self.chunk_segment_length = chunk_segment_length
        self.step_size = int((1 - chunk_overlap) * chunk_segment_length)
        self.max_shift = max_shift
        self.shift_offset = shift_offset
        self.current_offset = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_offset < self.shift_length:
            inner_offset = self.current_offset
            chunk_offset = self.current_offset - self.max_shift + self.shift_offset
            chunk_length = min(self.length - self.current_offset, self.chunk_segment_length)

            self.current_offset += self.step_size
            return inner_offset, chunk_offset, chunk_length
        else:
            raise StopIteration

def run_model(
    model,
    mix: torch.Tensor,
    shifts: int = 1,
    chunk: bool = True,
    chunk_overlap: float = 0.25,
    chunk_transition_power: float = 1.0,
    chunk_segment: float = 39/5,
):
    max_shift = int(0.5 * model.samplerate)
    chunk_segment_length: int = int(model.samplerate * chunk_segment)
    chunk_weight = torch.cat(
        [
            torch.arange(1, chunk_segment_length // 2 + 1),
            torch.arange(
                chunk_segment_length - chunk_segment_length // 2, 0, -1
            ),
        ]
    ).to(mix.device)
    chunk_weight = (chunk_weight / chunk_weight.max()) ** chunk_transition_power

    batch, channels, length = mix.shape
    out = torch.zeros((batch, len(model.sources), channels, length)).to(mix.device)

    for shift_offset, shift_length in Shifts(length, max_shift, shifts):

        chunk_out = torch.zeros(batch, len(model.sources), channels, shift_length).to(mix.device)
        chunk_sum_weight = torch.zeros(shift_length).to(mix.device)

        for inner_offset, chunk_offset, chunk_length in Chunks(length, max_shift, shift_offset, shift_length, chunk_overlap, chunk_segment_length):

            chunk_padded_mix = pad_symmetrically(mix, chunk_segment_length, offset=chunk_offset, length=chunk_length).to(mix.device)

            model_out = model(chunk_padded_mix)

            model_out = center_trim(model_out, chunk_length)

            chunk_out[..., inner_offset : inner_offset + chunk_length] += chunk_weight[:chunk_length] * model_out
            chunk_sum_weight[inner_offset : inner_offset + chunk_length] += chunk_weight[:chunk_length]
        chunk_out /= chunk_sum_weight

        out += chunk_out[..., max_shift - shift_offset : shift_length + max_shift - shift_offset]

    out /= shifts

    return out

def postprocess(output: torch.Tensor, ref: torch.Tensor):
    # un-normalized
    output *= ref.std() + 1e-8
    output += ref.mean()

    return dict(zip(model.sources, output[0]))

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    filename = "test2.mp3"
    device = "mps"

    model = torch.load("models/htdemucs.pt").models[0].to(device)
    wav = load_audio(model, filename).to(device)

    ref, mix = preprocess(wav, model.samplerate, model.samplerate, model.audio_channels)
    output = run_model(
        model,
        mix,
        shifts=2,
        chunk=True,
        chunk_overlap=0.25,
        chunk_transition_power=1.0,
        chunk_segment=model.segment,
    )

    separated = postprocess(output, ref)

    for stem, audio_data in separated.items():
        save_audio(
            audio_data,
            f"separated/coreml/{stem}_{filename}",
            samplerate=model.samplerate,
        )
