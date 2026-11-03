"""A1 compression primitives: masks, weight-only quantization, QAT and storage.

Inference uses dequantized floating-point kernels. Packed storage is real; low-bit
arithmetic acceleration is NOT claimed. Activations and batch norm remain FP32.
"""
import copy
import math
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune, parametrize

BITS = (8, 4, 2, 1)
SPARSITIES = (.1, .3, .5)


def layers(model):
    """Only conv/linear weights are compressible; BN and biases stay FP32."""
    return {n: m for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))}


def magnitude_mask(w, sparsity):
    """Remove exactly floor(s*N) entries, including ties, deterministically."""
    mask = torch.ones_like(w).flatten()
    k = int(w.numel() * sparsity)
    if k:
        idx = torch.argsort(w.detach().abs().flatten(), stable=True)[:k]
        mask[idx] = 0
    return mask.reshape_as(w)


def codes_and_scale(w, bits, mask):
    """Per-output-channel symmetric quantizer. Q1 is masked binary +/-alpha.

    For b>=2, codes 0..2*qmax represent signed levels -qmax..qmax.
    Q1 needs an explicit pruning mask because its two levels exclude zero.
    Scales are recalculated from current master weights during QAT.
    """
    dims = tuple(range(1, w.ndim))
    if bits == 1:
        scale = (w.abs()*mask).sum(dims, keepdim=True)
        scale = scale / mask.sum(dims, keepdim=True).clamp_min(1)
        code = (w >= 0).to(torch.uint8)
        value = (2*code.to(w.dtype)-1)*scale*mask
    else:
        qmax = 2**(bits-1)-1
        scale = (w*mask).abs().amax(dims, keepdim=True).clamp_min(1e-12)/qmax
        signed = torch.round((w*mask)/scale).clamp(-qmax, qmax)
        code = (signed+qmax).to(torch.uint8)
        value = signed*scale*mask
    return code, scale, value


class Compression(nn.Module):
    """Fixed pruning mask + straight-through quantizer over FP32 master weights."""
    def __init__(self, weight, bits, sparsity):
        super().__init__()
        self.bits = bits
        self.register_buffer('mask', magnitude_mask(weight, sparsity))

    def forward(self, w):
        masked = w*self.mask
        if self.bits == 32:
            return masked
        if self.bits == 16:
            value = masked.half().float()
        else:
            _, _, value = codes_and_scale(w, self.bits, self.mask)
        # Forward is quantized, backward derivative is the fixed mask.
        return masked + (value-masked).detach()


def attach(model, config):
    """Attach once; update_config handles later adaptive reallocation."""
    for name, module in layers(model).items():
        bits, sparsity = config.get(name, (32, 0.))
        parametrize.register_parametrization(module, 'weight',
                                             Compression(module.weight, bits, sparsity))
    return model


def update_config(model, config):
    for name, module in layers(model).items():
        c = module.parametrizations.weight[0]
        w = module.parametrizations.weight.original
        c.bits, sparsity = config[name]
        c.mask.copy_(magnitude_mask(w, sparsity))


def materialize(model):
    """Make a plain model with the effective weights, for export/histograms."""
    result = copy.deepcopy(model)
    # Rebuild plain modules. Removing parametrizations from a deep copy can
    # mutate a shared generated Python class and break the original model.
    for name, m in list(layers(result).items()):
        if not parametrize.is_parametrized(m, 'weight'):
            continue
        if isinstance(m, nn.Conv2d):
            plain = nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size,
                             m.stride, m.padding, m.dilation, m.groups,
                             m.bias is not None, m.padding_mode)
        else:
            plain = nn.Linear(m.in_features, m.out_features, m.bias is not None)
        plain = plain.to(device=m.weight.device, dtype=m.weight.dtype)
        with torch.no_grad():
            plain.weight.copy_(m.weight)
            if m.bias is not None:
                plain.bias.copy_(m.bias)
        plain.train(m.training)
        parent_name, _, child = name.rpartition('.')
        parent = result.get_submodule(parent_name) if parent_name else result
        setattr(parent, child, plain)
    return result


def uniform(model, bits, sparsity=0.):
    return {n: (bits, sparsity) for n in layers(model)}


def pruned_copy(model, sparsity, structured=False):
    """PyTorch global L1 or output-filter L2 masking.

    Structured here means whole-filter masking, NOT graph surgery. ResNet skip
    connections make physical channel removal more involved. Dense shape stays.
    """
    result = copy.deepcopy(model)
    modules = list(layers(result).values())
    if structured:
        for m in modules:
            prune.ln_structured(m, 'weight', amount=sparsity, n=2, dim=0)
    else:
        prune.global_unstructured([(m, 'weight') for m in modules],
                                  pruning_method=prune.L1Unstructured, amount=sparsity)
    for m in modules:
        prune.remove(m, 'weight')
    return result


def pack_codes(codes, bits):
    """Little bit-order packing for 1/2/4/8-bit unsigned integer codes."""
    a = np.asarray(codes, dtype=np.uint8).reshape(-1)
    shifts = np.arange(bits, dtype=np.uint8)
    return np.packbits(((a[:, None] >> shifts) & 1).reshape(-1), bitorder='little').tobytes()


def unpack_codes(payload, bits, count):
    bit = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='little')[:count*bits]
    return (bit.reshape(-1, bits)*(1 << np.arange(bits))).sum(1).astype(np.uint8)


def encode_tensor(w, bits, mask=None):
    """Choose the smaller of dense encoding and bitmap+nonzero encoding.

    Accounts for actual bit packing, scales, bitmap, shape and metadata when
    serialized. Does not assume that sparsity alone shrinks a dense checkpoint.
    """
    w = w.detach().cpu().float()
    mask = torch.ones_like(w) if mask is None else mask.detach().cpu()
    if bits in BITS:
        code, scale, value = codes_and_scale(w, bits, mask)
        a = code.numpy().reshape(-1)
        # Q1 needs a zero bitmap even in dense mode if weights were pruned.
        active = mask.numpy().reshape(-1).astype(bool) if bits == 1 else value.numpy().reshape(-1) != 0
        dense = {'kind': 'dense', 'data': pack_codes(a, bits)}
        if bits == 1 and not active.all():
            dense['mask'] = np.packbits(active, bitorder='little').tobytes()
        sparse = {'kind': 'sparse', 'data': pack_codes(a[active], bits),
                  'mask': np.packbits(active, bitorder='little').tobytes(), 'nnz': int(active.sum())}
        record = min((dense, sparse), key=lambda x: len(pickle.dumps(x, protocol=4)))
        record.update(scale=scale.numpy(), bits=bits, shape=tuple(w.shape))
    else:
        dtype = np.float16 if bits == 16 else np.float32
        a = (w*mask).numpy().astype(dtype).reshape(-1)
        active = a != 0
        dense = {'kind': 'dense', 'data': a.tobytes()}
        sparse = {'kind': 'sparse', 'data': a[active].tobytes(),
                  'mask': np.packbits(active, bitorder='little').tobytes(), 'nnz': int(active.sum())}
        record = min((dense, sparse), key=lambda x: len(pickle.dumps(x, protocol=4)))
        record.update(bits=bits, shape=tuple(w.shape))
    return record


def decode_tensor(r):
    n = math.prod(r['shape'])
    active = (np.unpackbits(np.frombuffer(r['mask'], np.uint8), bitorder='little')[:n].astype(bool)
              if 'mask' in r else np.ones(n, bool))
    count = int(active.sum()) if r['kind'] == 'sparse' else n
    if r['bits'] in BITS:
        code = unpack_codes(r['data'], r['bits'], count)
        fill = 0 if r['bits'] == 1 else 2**(r['bits']-1)-1
        full = np.full(n, fill, np.uint8)
        if r['kind'] == 'sparse':
            full[active] = code
        else:
            full[:] = code
        signed = 2*full.astype(np.float32)-1 if r['bits'] == 1 else full.astype(np.float32)-fill
        value = signed.reshape(r['shape'])*r['scale']
        value = value*active.reshape(r['shape'])
    else:
        dtype = np.float16 if r['bits'] == 16 else np.float32
        a = np.frombuffer(r['data'], dtype=dtype)
        value = np.zeros(n, np.float32)
        if r['kind'] == 'sparse':
            value[active] = a
        else:
            value[:] = a
        value = value.reshape(r['shape'])
    return torch.from_numpy(value.copy()).float()


def archive(model):
    """Portable CPU archive. Pickle files must only be loaded from trusted sources."""
    plain = materialize(model)
    state = {n: t.detach().cpu() for n, t in plain.state_dict().items()}
    records = {}
    for name, m in layers(model).items():
        if parametrize.is_parametrized(m, 'weight'):
            c = m.parametrizations.weight[0]
            records[name+'.weight'] = encode_tensor(m.parametrizations.weight.original, c.bits, c.mask)
        else:
            records[name+'.weight'] = encode_tensor(m.weight, 32)
        del state[name+'.weight']
    return {'format': 1, 'weights': records, 'other': state}


def archive_bytes(model):
    return pickle.dumps(archive(model), protocol=4)


def load_archive(model, path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)  # trusted local experiment artifact only
    state = obj['other']
    state.update({n: decode_tensor(r) for n, r in obj['weights'].items()})
    model.load_state_dict(state)
    return model


def estimate_layer_bytes(weight, bits, sparsity):
    """Conservative bitmap encoding bound; dense alternative may be smaller."""
    n = weight.numel()
    nnz = n-int(n*sparsity)
    scale = weight.shape[0]*4
    return math.ceil(n/8)+math.ceil(nnz*bits/8)+scale+128


def pareto(rows, size='size_mb', acc='test_accuracy'):
    """Minimize size, maximize accuracy. Keep each strict improvement."""
    out, best = [], -float('inf')
    for r in sorted(rows, key=lambda r: (r[size], -r[acc])):
        if r[acc] > best:
            out.append(r)
            best = r[acc]
    return out


def hypervolume(rows, reference_mb, accuracy_floor=0.):
    """2D dominated area, in MB * percentage points, fixed shared reference."""
    front = pareto([r for r in rows if r['size_mb'] < reference_mb
                    and r['test_accuracy'] > accuracy_floor])
    area = 0.
    for i, r in enumerate(front):
        right = front[i+1]['size_mb'] if i+1 < len(front) else reference_mb
        area += (right-r['size_mb'])*(r['test_accuracy']-accuracy_floor)
    return area
