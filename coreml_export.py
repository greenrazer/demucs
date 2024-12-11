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


def _replace_dict(_dict, *subs) -> dict:
    if _dict is None:
        _dict = {}
    else:
        _dict = copy.copy(_dict)
    for key, value in subs:
        _dict[key] = value
    return _dict


class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, _dict, *args, **kwargs):
            self.func = func
            self._dict = _dict
            self.args = args
            self.kwargs = kwargs

        def result(self):
            if self._dict["run"]:
                return self.func(*self.args, **self.kwargs)
            else:
                raise Exception("placeholder")

    def __init__(self, workers=0):
        self._dict = {"run": True}

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, self._dict, *args, **kwargs)

    def shutdown(self, *_, **__):
        self._dict["run"] = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return


def pad_symmetrically(tensor, target_length, offset=0, length=None):
    total_length = tensor.shape[-1]

    if length is None:
        length = total_length - offset
    else:
        length = min(total_length - offset, length)

    delta = target_length - length
    total_length = tensor.shape[-1]
    assert delta >= 0

    start = offset - delta // 2
    end = start + target_length

    correct_start = max(0, start)
    correct_end = min(total_length, end)

    pad_left = correct_start - start
    pad_right = end - correct_end

    out = F.pad(tensor[..., correct_start:correct_end], (pad_left, pad_right))
    return out


def center_trim(tensor: torch.Tensor, reference: Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
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


def apply_model(
    model,
    mix,
    tensor_offset=0,
    tensor_length=None,
    segment: Optional[float] = None,
) -> torch.Tensor:
    if tensor_length is None:
        tensor_length = mix.shape[-1]

    batch, channels, length = mix.shape
    out = torch.zeros(batch, len(model.sources), channels, tensor_length).to(mix.device)

    valid_length: int
    if segment is not None:
        valid_length = int(segment * model.samplerate)
    elif hasattr(model, "valid_length"):
        valid_length = model.valid_length(tensor_length)
    else:
        valid_length = tensor_length

    padded_mix = pad_symmetrically(
        mix, valid_length, offset=tensor_offset, length=tensor_length
    ).to(mix.device)

    with torch.no_grad():
        out = model(padded_mix)
    return center_trim(out, tensor_length)


def apply_model_split(
    model,
    mix,
    tensor_offset=0,
    tensor_length=None,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    segment: Optional[float] = None,
):
    batch, channels, length = mix.shape

    out = torch.zeros(batch, len(model.sources), channels, tensor_length).to(mix.device)
    sum_weight = torch.zeros(tensor_length).to(mix.device)

    if segment is None:
        segment = model.segment

    segment_length: int = int(model.samplerate * segment)

    weight = torch.cat(
        [
            torch.arange(1, segment_length // 2 + 1),
            torch.arange(segment_length - segment_length // 2, 0, -1),
        ]
    ).to(mix.device)

    weight = (weight / weight.max()) ** transition_power
    for inner_offset in range(0, tensor_length, int((1 - overlap) * segment_length)):
        model_out = apply_model(
            model,
            mix,
            tensor_offset=tensor_offset + inner_offset,
            tensor_length=min(tensor_length - inner_offset, segment_length),
        )
        chunk_length = model_out.shape[-1]
        out[..., inner_offset : inner_offset + segment_length] += (
            weight[:chunk_length] * model_out
        )
        sum_weight[inner_offset : inner_offset + segment_length] += weight[
            :chunk_length
        ]
    out /= sum_weight
    return out


def apply_model_shifts(
    model,
    mix,
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
):
    batch, channels, length = mix.shape
    out = torch.zeros((batch, len(model.sources), channels, length)).to(mix.device)

    max_shift = int(0.5 * model.samplerate)
    padded_mix = pad_symmetrically(mix, length + 2 * max_shift).to(mix.device)

    for shift_idx in range(shifts):
        offset = max_shift // (shift_idx + 1)

        if split:
            res = apply_model_split(
                model,
                padded_mix,
                tensor_offset=offset,
                tensor_length=length + max_shift - offset,
                overlap=overlap,
                transition_power=transition_power,
            )
        else:
            res = apply_model(
                model,
                padded_mix,
                tensor_offset=offset,
                tensor_length=length + max_shift - offset,
            )

        out += res[..., max_shift - offset : length + max_shift - offset]
    out /= shifts
    return out


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


def separate_tensor(
    model, wav: torch.Tensor, input_sample_rate: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    sample_rate = model.samplerate
    audio_channels = model.audio_channels

    if input_sample_rate is not None and input_sample_rate != sample_rate:
        wav = convert_audio_channels(wav, audio_channels)
        wav = julius.resample_frac(wav, input_sample_rate, sample_rate)

    # normalize
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8

    device = "mps"
    mix = wav.unsqueeze(0)
    transition_power = 1.0
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."

    batch, channels, length = mix.shape
    out = torch.zeros((batch, len(model.sources), channels, length)).to(device)

    totals = [0.0] * len(model.sources)
    for sub_model, model_weights in zip(model.models, model.weights):
        sub_model.to(device)

        res = apply_model_shifts(
            sub_model,
            mix,
            shifts=1,
            split=True,
            overlap=0.25,
            transition_power=transition_power,
        )

        for k, inst_weight in enumerate(model_weights):
            res[:, k, :, :] *= inst_weight
            totals[k] += inst_weight
        out += res

    for k in range(out.shape[1]):
        out[:, k, :, :] /= totals[k]

    # un-normalized
    out *= ref.std() + 1e-8
    out += ref.mean()
    wav *= ref.std() + 1e-8
    wav += ref.mean()

    return (wav, dict(zip(model.sources, out[0])))


def _load_audio(model, track: Path):
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

def prevent_clip(wav, mode='rescale'):
    """
    different strategies for avoiding raw clipping.
    """
    if mode is None or mode == 'none':
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
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

def save_audio(wav: torch.Tensor,
               path: Union[str, Path],
               samplerate: int,
               bitrate: int = 320,
               clip: Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
               bits_per_sample: Literal[16, 24, 32] = 16,
               as_float: bool = False,
               preset: Literal[2, 3, 4, 5, 6, 7] = 2):
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        encode_mp3(wav, path, samplerate, bitrate, preset, verbose=True)
    elif suffix == ".wav":
        if as_float:
            bits_per_sample = 32
            encoding = 'PCM_F'
        else:
            encoding = 'PCM_S'
        torchaudio.save(str(path), wav, sample_rate=samplerate,
                encoding=encoding, bits_per_sample=bits_per_sample)
    elif suffix == ".flac":
        torchaudio.save(str(path), wav, sample_rate=samplerate, bits_per_sample=bits_per_sample)
    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")

if __name__ == "__main__":
    filename = "test2.mp3"

    model = torch.load("models/htdemucs.pt")

    wav, separated = separate_tensor(
        model, _load_audio(model, filename).to("mps"), model.samplerate
    )

    for stem, audio_data in separated.items():
        save_audio(
            audio_data,
            f"separated/coreml/{stem}_{filename}",
            samplerate=model.samplerate,
        )
