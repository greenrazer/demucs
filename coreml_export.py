import demucs.api
from demucs.api import AudioFile

import copy
from pathlib import Path
import random
from threading import Lock
from typing import Callable, Dict, Optional, Tuple, Union
import torch
from torch.nn import functional as F
import subprocess
import torchaudio
import julius


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


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, torch.Tensor)
        return TensorChunk(tensor_or_chunk)


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
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress: bool = False,
    device=None,
    num_workers: int = 0,
    segment: Optional[float] = None,
    pool=None,
    lock=None,
    callback: Optional[Callable[[dict], None]] = None,
    callback_arg: Optional[dict] = None,
) -> torch.Tensor:
    pool = DummyPoolExecutor()
    lock = Lock()
    kwargs = {
        "shifts": shifts,
        "split": split,
        "overlap": overlap,
        "transition_power": transition_power,
        "progress": progress,
        "device": device,
        "pool": pool,
        "segment": segment,
        "lock": lock,
    }

    out: Union[float, torch.Tensor]
    res: Union[float, torch.Tensor]

    if "models" not in callback_arg:
        callback_arg["models"] = 1

    model.to(device)
    model.eval()

    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape

    valid_length: int
    if segment is not None:
        valid_length = int(segment * model.samplerate)
    elif hasattr(model, "valid_length"):
        valid_length = model.valid_length(length)  # type: ignore
    else:
        valid_length = length
    mix = tensor_chunk(mix)
    assert isinstance(mix, TensorChunk)
    padded_mix = mix.padded(valid_length).to(device)
    with lock:
        if callback is not None:
            callback(_replace_dict(callback_arg, ("state", "start")))  # type: ignore
    with torch.no_grad():
        out = model(padded_mix)
    with lock:
        if callback is not None:
            callback(_replace_dict(callback_arg, ("state", "end")))  # type: ignore
    assert isinstance(out, torch.Tensor)
    return center_trim(out, length)


def apply_model_split(
    model,
    mix,
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress: bool = False,
    device=None,
    num_workers: int = 0,
    segment: Optional[float] = None,
    pool=None,
    lock=None,
    callback: Optional[Callable[[dict], None]] = None,
    callback_arg: Optional[dict] = None,
):
    pool = DummyPoolExecutor()
    lock = Lock()
    kwargs = {
        "shifts": shifts,
        "split": split,
        "overlap": overlap,
        "transition_power": transition_power,
        "progress": progress,
        "device": device,
        "pool": pool,
        "segment": segment,
        "lock": lock,
    }

    out: Union[float, torch.Tensor]
    res: Union[float, torch.Tensor]

    if "models" not in callback_arg:
        callback_arg["models"] = 1

    model.to(device)
    model.eval()

    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape

    kwargs["split"] = False
    out = torch.zeros(batch, len(model.sources), channels, length, device=mix.device)
    sum_weight = torch.zeros(length, device=mix.device)
    if segment is None:
        segment = model.segment
    assert segment is not None and segment > 0.0
    segment_length: int = int(model.samplerate * segment)
    stride = int((1 - overlap) * segment_length)
    offsets = range(0, length, stride)
    scale = float(format(stride / model.samplerate, ".2f"))
    # We start from a triangle shaped weight, with maximal weight in the middle
    # of the segment. Then we normalize and take to the power `transition_power`.
    # Large values of transition power will lead to sharper transitions.
    weight = torch.cat(
        [
            torch.arange(1, segment_length // 2 + 1, device=device),
            torch.arange(segment_length - segment_length // 2, 0, -1, device=device),
        ]
    )
    assert len(weight) == segment_length
    # If the overlap < 50%, this will translate to linear transition when
    # transition_power is 1.
    weight = (weight / weight.max()) ** transition_power
    futures = []
    for offset in offsets:
        chunk = TensorChunk(mix, offset, segment_length)
        future = pool.submit(
            apply_model,
            model,
            chunk,
            **kwargs,
            callback_arg=callback_arg,
            callback=(
                lambda d, i=offset: callback(_replace_dict(d, ("segment_offset", i)))
                if callback
                else None
            ),
        )
        futures.append((future, offset))
        offset += segment_length
    for future, offset in futures:
        try:
            chunk_out = future.result()  # type: torch.Tensor
        except Exception:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        chunk_length = chunk_out.shape[-1]
        out[..., offset : offset + segment_length] += (
            weight[:chunk_length] * chunk_out
        ).to(mix.device)
        sum_weight[offset : offset + segment_length] += weight[:chunk_length].to(
            mix.device
        )
    assert sum_weight.min() > 0
    out /= sum_weight
    assert isinstance(out, torch.Tensor)
    return out


def apply_model_shifts(
    model,
    mix,
    shifts: int = 1,
    split: bool = True,
    overlap: float = 0.25,
    transition_power: float = 1.0,
    progress: bool = False,
    device=None,
    num_workers: int = 0,
    segment: Optional[float] = None,
    pool=None,
    lock=None,
    callback: Optional[Callable[[dict], None]] = None,
    callback_arg: Optional[dict] = None,
):
    pool = DummyPoolExecutor()
    lock = Lock()
    kwargs = {
        "shifts": shifts,
        "split": split,
        "overlap": overlap,
        "transition_power": transition_power,
        "progress": progress,
        "device": device,
        "pool": pool,
        "segment": segment,
        "lock": lock,
    }

    out: Union[float, torch.Tensor]
    res: Union[float, torch.Tensor]

    if "models" not in callback_arg:
        callback_arg["models"] = 1

    model.to(device)
    model.eval()

    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape

    kwargs["shifts"] = 0
    max_shift = int(0.5 * model.samplerate)
    mix = tensor_chunk(mix)
    assert isinstance(mix, TensorChunk)
    padded_mix = mix.padded(length + 2 * max_shift)
    out = 0.0
    for shift_idx in range(shifts):
        offset = random.randint(0, max_shift)
        shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
        kwargs["callback"] = None

        if split:
            res = apply_model_split(model, shifted, **kwargs, callback_arg=callback_arg)
        else:
            res = apply_model(model, shifted, **kwargs, callback_arg=callback_arg)

        shifted_out = res
        out += shifted_out[..., max_shift - offset :]
    out /= shifts
    assert isinstance(out, torch.Tensor)
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
        # wav = convert_audio(wav, sr, sample_rate, audio_channels)
        wav = convert_audio_channels(wav, audio_channels)
        wav = julius.resample_frac(wav, input_sample_rate, sample_rate)

    # normalize
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8

    device = "mps"
    mix = wav.unsqueeze(0)

    out = torch.zeros((mix.shape[0], 4, mix.shape[1], mix.shape[2]))
    totals = [0.0] * len(model.sources)
    for sub_model, model_weights in zip(model.models, model.weights):
        sub_model.to(device)

        res = apply_model_shifts(
            sub_model,
            mix,
            segment=None,
            shifts=1,
            split=True,
            overlap=0.25,
            device=device,
            num_workers=0,
            callback=None,
            callback_arg={
                "audio_length": wav.shape[1],
                "model_idx_in_bag": 0,
                "shift_idx": 0,
                "segment_offset": 0,
            },
            progress=False,
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


if __name__ == "__main__":
    filename = "test2.mp3"

    model = torch.load("models/htdemucs.pt")

    wav, separated = separate_tensor(
        model, _load_audio(model, filename), model.samplerate
    )

    for stem, audio_data in separated.items():
        demucs.api.save_audio(
            audio_data,
            f"separated/coreml/{stem}_{filename}",
            samplerate=model.samplerate,
        )
