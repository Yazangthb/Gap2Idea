"""Bounded-memory safetensors loading for the tested Windows torch stack.

Read one parameter at a time using ordinary file I/O. Each tensor is decoded by
the official safetensors parser, then copied into an already allocated model.
This avoids large memory-mapped storage slicing (a native access violation seen
on Windows with torch 2.8 / transformers 4.53 / safetensors 0.6).
"""
import json
import struct


def copy_safetensors(model, path):
    import torch
    from safetensors.torch import load
    target = model.state_dict()
    with open(path, "rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors file")
        header_length = struct.unpack("<Q", raw)[0]
        if not 0 < header_length <= 100_000_000:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(header_length))
        metadata = {k: v for k, v in header.items() if k != "__metadata__"}
        if set(metadata) != set(target):
            raise ValueError(f"Checkpoint/model keys differ: missing={set(target)-set(metadata)}, unexpected={set(metadata)-set(target)}")
        data_start = 8 + header_length
        size = stream.seek(0, 2) - data_start
        previous_end = 0
        for name, desc in sorted(metadata.items(), key=lambda item: item[1]["data_offsets"][0]):
            begin, end = desc["data_offsets"]
            if begin != previous_end or not begin <= end <= size:
                raise ValueError("Invalid or non-contiguous safetensors offsets")
            previous_end = end
            stream.seek(data_start + begin)
            contents = stream.read(end - begin)
            if len(contents) != end - begin:
                raise ValueError("Truncated safetensors parameter")
            # A one-tensor safetensors container lets the official library check
            # dtype, dimensions and byte count without mapping the original file.
            one_header = json.dumps({"tensor": {"dtype": desc["dtype"], "shape": desc["shape"],
                                               "data_offsets": [0, end - begin]}}, separators=(",", ":")).encode()
            one_header += b" " * (-len(one_header) % 8)
            tensor = load(struct.pack("<Q", len(one_header)) + one_header + contents)["tensor"]
            if tuple(tensor.shape) != tuple(target[name].shape):
                raise ValueError(f"Checkpoint shape differs for {name}")
            with torch.no_grad():
                target[name].copy_(tensor)
            del tensor, contents
        if previous_end != size:
            raise ValueError("Trailing safetensors data")


def load_windows_model(path, config, device, dtype):
    import torch
    from transformers import AutoModelForSequenceClassification
    # Allocate directly on the final device. This correctly initializes
    # non-persistent rotary buffers as well as avoiding full CPU weight copies.
    with torch.device(device):
        model = AutoModelForSequenceClassification.from_config(
            config, torch_dtype=dtype, attn_implementation="eager")
    copy_safetensors(model, path)
    return model.eval()
