"""Run all seven A1 phases; see README.md for commands and interpretation."""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from compression import (layers, attach, uniform, pruned_copy, materialize, archive_bytes,
                         load_archive, estimate_layer_bytes, pareto, hypervolume,
                         update_config, BITS, SPARSITIES)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TinyNet(nn.Module):
    """Synthetic verification only; never presented as CIFAR-10 results."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        return self.fc(self.pool(self.relu(self.conv(x))).flatten(1))


def make_model(smoke=False):
    if smoke:
        return TinyNet()
    from torchvision.models import resnet18
    model = resnet18(weights=None, num_classes=10)
    # CIFAR images are 32x32; preserve resolution with a 3x3 stem.
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def loaders(args):
    generator = torch.Generator().manual_seed(args.seed)
    if args.smoke:
        def dataset(n):
            x = torch.randn(n, 3, 8, 8, generator=generator)
            # Artificial, learnable labels, unrelated to CIFAR-10.
            y = ((x[:, 0].mean((1, 2)) > 0).long())
            return TensorDataset(x, y)
        train, valid, test = dataset(32), dataset(16), dataset(16)
        probe = valid
        split = {'mode': 'SYNTHETIC VERIFICATION ONLY', 'train': 32, 'validation': 16, 'test': 16}
    else:
        from torchvision import datasets, transforms
        mean, std = (.4914, .4822, .4465), (.247, .2435, .2616)
        augment = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        clean = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        raw_train = datasets.CIFAR10(args.data, train=True, download=args.download, transform=augment)
        raw_valid = datasets.CIFAR10(args.data, train=True, download=args.download, transform=clean)
        test = datasets.CIFAR10(args.data, train=False, download=args.download, transform=clean)
        order = torch.randperm(50000, generator=generator).tolist()
        train, valid = Subset(raw_train, order[5000:]), Subset(raw_valid, order[:5000])
        probe = Subset(raw_valid, order[:min(args.probe_samples, 5000)])
        split = {'mode': 'CIFAR-10', 'train_indices': order[5000:], 'validation_indices': order[:5000],
                 'test': 10000, 'probe': len(probe)}
    (args.out/'split.json').write_text(json.dumps(split))
    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, generator=generator)
    return loader(train, True), loader(valid), loader(probe), loader(test)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return {'accuracy': 100*correct/count, 'loss': loss/count}


def train(model, trainloader, validloader, epochs, lr, device, freeze_bn=False):
    """Validation selects the best epoch; test labels are never consulted."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best = copy.deepcopy(model.state_dict())
    best_acc = evaluate(model, validloader, device)['accuracy']
    history = []
    for epoch in range(epochs):
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        running, count = 0., 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()*len(y)
            count += len(y)
        sched.step()
        score = evaluate(model, validloader, device)
        history.append({'epoch': epoch+1, 'train_loss': running/count, **score})
        print(f'epoch {epoch+1}/{epochs}: val={score["accuracy"]:.2f}%', flush=True)
        if score['accuracy'] >= best_acc:
            best_acc = score['accuracy']
            best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return history


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def histograms(model, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    weights = [(n, m.weight.detach().cpu().numpy().ravel()) for n, m in layers(model).items()]
    fig, axes = plt.subplots(int(np.ceil((len(weights)+1)/4)), 4,
                             figsize=(14, 2.5*int(np.ceil((len(weights)+1)/4))), squeeze=False)
    all_weights = np.concatenate([w for _, w in weights])
    for ax, (name, values) in zip(axes.flat, [('Overall', all_weights)]+weights):
        ax.hist(values, bins=70, color='#225b92', log=True)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Weight'); ax.set_ylabel('Count (log)')
    for ax in list(axes.flat)[len(weights)+1:]:
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(path, dpi=140); plt.close(fig)


def measured_latency(model, loader, device, repeats):
    """Dequantized floating-point latency at the recorded batch size, not INT speed."""
    x = next(iter(loader))[0].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
    return 1000*(time.perf_counter()-start)/repeats, len(x)


def probe_candidates(master, probe, device, round_id):
    """Independently compress each layer for all 12 joint bit/sparsity choices.

    No test-set tuning. Store signed accuracy changes plus CE-loss changes; CE
    provides a tie-breaker when accuracy resolution is coarse on small probes.
    """
    reference = evaluate(master, probe, device)
    rows = []
    for name, m in layers(master).items():
        for bits in BITS:
            for sparsity in SPARSITIES:
                candidate = attach(copy.deepcopy(master), {name: (bits, sparsity)})
                score = evaluate(candidate, probe, device)
                rows.append({'round': round_id, 'layer': name, 'bits': bits, 'sparsity': sparsity,
                             'delta_accuracy': reference['accuracy']-score['accuracy'],
                             'delta_loss': score['loss']-reference['loss'],
                             'estimated_bytes': estimate_layer_bytes(m.weight, bits, sparsity)})
                del candidate
        print(f'probed {name}, round {round_id}', flush=True)
    return rows


def allocate(model, rows, target_ratio, tau):
    """Sensitivity-guided discrete multiple-choice byte allocator.

    exp(-delta/tau) is interpreted as COMPRESSION demand: insensitive layers
    receive more compression, rather than fewer retained bits for sensitive
    layers. Each layer must choose a feasible (bits,sparsity) pair. Start at
    minimum bytes, spend spare budget on the largest predicted loss reduction
    per additional byte. A local heuristic, not an optimality guarantee.
    """
    modules = layers(model)
    grouped = {n: [r for r in rows if r['layer'] == n] for n in modules}
    sensitivity = {n: max(0., next(r['delta_accuracy'] for r in rr
                                 if r['bits'] == 4 and r['sparsity'] == .3)) for n, rr in grouped.items()}
    demand = {n: np.exp(-d/tau) for n, d in sensitivity.items()}
    total_demand = sum(demand.values())
    # Scores measured in validation CE. Accuracy sensitivity adjusts priority.
    for n, rr in grouped.items():
        for r in rr:
            r['cost'] = max(0., r['delta_loss']) / max(demand[n], 1e-9)
    choices = {n: min(rr, key=lambda r: (r['estimated_bytes'], r['cost'])) for n, rr in grouped.items()}
    weight_budget = sum(m.weight.numel()*4 for m in modules.values())/target_ratio
    minimum = sum(r['estimated_bytes'] for r in choices.values())
    if minimum > weight_budget:
        raise ValueError(f'Target {target_ratio}x infeasible: minimum estimated bytes={minimum}, budget={weight_budget}')
    while True:
        used = sum(r['estimated_bytes'] for r in choices.values())
        upgrades = []
        for n, rr in grouped.items():
            current = choices[n]
            for r in rr:
                extra = r['estimated_bytes']-current['estimated_bytes']
                gain = current['cost']-r['cost']
                if extra > 0 and gain > 0 and used+extra <= weight_budget:
                    upgrades.append((gain/extra, n, r))
                elif extra <= 0 and gain > 0:
                    upgrades.append((float('inf'), n, r))
        if not upgrades: break
        _, name, choice = max(upgrades, key=lambda item: item[0])
        choices[name] = choice
    config = {n: (r['bits'], r['sparsity']) for n, r in choices.items()}
    detail = [{'layer': n, 'bits': r['bits'], 'sparsity': r['sparsity'],
               'estimated_bytes': r['estimated_bytes'], 'compression_demand': demand[n]/total_demand,
               'sensitivity': sensitivity[n], 'weight_budget_bytes': weight_budget}
              for n, r in choices.items()]
    return config, detail


def plots(rows, probe_rows, output, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def line(sub, x, filename, xlabel):
        if not sub: return
        sub = sorted(sub, key=lambda r: r[x])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r[x] for r in sub], [r['test_accuracy'] for r in sub], 'o-')
        ax.set(xlabel=xlabel, ylabel='Test top-1 accuracy (%)', title=label)
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/filename, dpi=160); plt.close(fig)
    line([r for r in rows if r['phase']=='unstructured'], 'requested_sparsity', 'accuracy_vs_sparsity.png', 'Requested sparsity')
    line([r for r in rows if r['phase']=='ptq'], 'bits', 'accuracy_vs_bits.png', 'Weight bits')
    fig, ax = plt.subplots(figsize=(9, 5))
    for phase in sorted({r['phase'] for r in rows}):
        sub = [r for r in rows if r['phase']==phase]
        ax.scatter([r['size_mb'] for r in sub], [r['test_accuracy'] for r in sub], label=phase)
    front = pareto(rows)
    ax.plot([r['size_mb'] for r in front], [r['test_accuracy'] for r in front], '--', color='black', label='Pareto')
    ax.set(xscale='log', xlabel='Actual packed archive size (MB, decimal)', ylabel='Test top-1 accuracy (%)', title=label)
    ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output/'accuracy_vs_size.png', dpi=160); plt.close(fig)
    sub = sorted([r for r in rows if r['phase']=='ptq'], key=lambda r:r['bits'])
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot([r['bits'] for r in sub], [r['size_mb'] for r in sub], 'o-')
    ax.set(xlabel='Weight bits', ylabel='Actual packed size (MB)', title=label)
    fig.tight_layout(); fig.savefig(output/'size_vs_bits.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for phase in ('unstructured','structured'):
        sub = sorted([r for r in rows if r['phase']==phase], key=lambda r:r['actual_weight_sparsity'])
        ax.plot([r['actual_weight_sparsity'] for r in sub], [r['test_accuracy'] for r in sub], 'o-', label=phase)
    ax.set(xlabel='Actual zero fraction', ylabel='Test accuracy (%)', title=label)
    ax.legend(); fig.tight_layout(); fig.savefig(output/'structured_vs_unstructured.png', dpi=160); plt.close(fig)
    # Common Q4 / 30% probe makes layer sensitivity directly comparable.
    sub = [r for r in probe_rows if r['round']==0 and r['bits']==4 and r['sparsity']==.3]
    fig, ax = plt.subplots(figsize=(9, max(3, len(sub)*.25)))
    ax.barh([r['layer'] for r in sub], [r['delta_accuracy'] for r in sub])
    ax.set(xlabel='Validation accuracy drop (percentage points)', title='Layer sensitivity: Q4 + 30% pruning')
    fig.tight_layout(); fig.savefig(output/'layer_sensitivity.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for phase in ('ptq', 'qat'):
        sub = sorted([r for r in rows if r['phase']==phase and r['bits'] in BITS], key=lambda r:r['bits'])
        ax.plot([r['bits'] for r in sub], [r['test_accuracy'] for r in sub], 'o-', label=phase)
    ax.set(xlabel='Weight bits', ylabel='Test accuracy (%)', title=label); ax.legend()
    fig.tight_layout(); fig.savefig(output/'ptq_vs_qat.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true', help='Tiny synthetic verification; no CIFAR claims')
    p.add_argument('--data', default='./data'); p.add_argument('--download', action='store_true')
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--finetune-epochs', type=int, default=10)
    p.add_argument('--rounds', type=int, default=3); p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--probe-samples', type=int, default=512); p.add_argument('--lr', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--tau', type=float, default=1.)
    p.add_argument('--targets', type=float, nargs='+', default=[4., 8., 16.])
    p.add_argument('--checkpoint', type=Path); p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if a.rounds < 2 or a.tau <= 0 or min(a.targets) <= 0:
        p.error('Use >=2 rounds, tau>0, targets>0.')
    if a.smoke:
        a.epochs, a.finetune_epochs, a.batch_size = 2, 1, 16
        a.targets = [1.5, 2.]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'plots').mkdir(exist_ok=True); (a.out/'models').mkdir(exist_ok=True)
    seed_everything(a.seed); torch.set_num_threads(a.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    meta = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    meta.update(torch_version=torch.__version__, device_used=str(device),
                quantization='weight-only simulation; FP32 arithmetic', size='actual pickle packed archive bytes')
    (a.out/'config.json').write_text(json.dumps(meta, indent=2))
    tr, va, probe, te = loaders(a)
    base = make_model(a.smoke).to(device)
    if a.checkpoint:
        base.load_state_dict(torch.load(a.checkpoint, map_location=device, weights_only=True))
        history = []
    else:
        history = train(base, tr, va, a.epochs, a.lr, device)
    torch.save(base.state_dict(), a.out/'models'/'baseline_fp32.pt')
    write_csv(a.out/'training.csv', history)
    histograms(base, a.out/'plots'/'baseline_histograms.png', 'Baseline weights')
    rows = []

    def record(model, name, phase, bits=32, sparsity=0., round_id=-1):
        validation = evaluate(model, va, device)
        test = evaluate(model, te, device)
        payload = archive_bytes(model)
        path = a.out/'models'/(name+'.pkl'); path.write_bytes(payload)
        # Verify that saved codes/masks reconstruct the weights used for scoring.
        restored = load_archive(make_model(a.smoke), path).to(device)
        x = next(iter(te))[0].to(device)
        with torch.no_grad():
            model.eval(); restored.eval()
            torch.testing.assert_close(model(x), restored(x), atol=2e-5, rtol=2e-4)
        latency, batch = measured_latency(restored, te, device, 5 if a.smoke else 30)
        plain = materialize(model)
        zero = sum((m.weight==0).sum().item() for m in layers(plain).values())
        count = sum(m.weight.numel() for m in layers(plain).values())
        row = {'name': name, 'phase': phase, 'bits': bits, 'requested_sparsity': sparsity,
               'actual_weight_sparsity': zero/count, 'round': round_id,
               'validation_accuracy': validation['accuracy'], 'test_accuracy': test['accuracy'],
               'test_loss': test['loss'], 'size_bytes': len(payload), 'size_mb': len(payload)/1e6,
               'fp_kernel_batch_ms': latency, 'latency_batch_size': batch}
        rows.append(row); write_csv(a.out/'results.csv', rows)
        print(f'{name}: test={test["accuracy"]:.2f}%, packed={len(payload)/1e6:.4f}MB', flush=True)
        return row

    record(base, 'baseline', 'baseline')
    # P2 / P3: compare global magnitude to filter masking at shared targets.
    for structured, sparsities in [(False, (.1,.3,.5,.7,.9)), (True, SPARSITIES)]:
        phase = 'structured' if structured else 'unstructured'
        for s in sparsities:
            model = pruned_copy(base, s, structured)
            name = f'{phase}_{int(s*100)}'
            record(model, name, phase, sparsity=s)
            histograms(model, a.out/'plots'/(name+'_hist.png'), name)
            del model
    # P4: FP16 is weight round-trip simulation; all compute remains FP32.
    for bits in (32,16,8,4,2,1):
        model = attach(copy.deepcopy(base), uniform(base, bits))
        record(model, f'ptq_q{bits}', 'ptq', bits)
        histograms(model, a.out/'plots'/f'ptq_q{bits}_hist.png', f'PTQ Q{bits}')
        del model
    # P5: one layer at a time, all 12 choices. Preserve complete evidence.
    probes = probe_candidates(base, probe, device, 0)
    write_csv(a.out/'sensitivity.csv', probes)
    # P6: each QAT run starts from the same FP32 baseline and same mask policy.
    for bits in BITS:
        seed_everything(a.seed)
        model = attach(copy.deepcopy(base), uniform(base, bits))
        h = train(model, tr, va, a.finetune_epochs, .001, device, freeze_bn=True)
        write_csv(a.out/f'qat_q{bits}_training.csv', h)
        record(model, f'qat_q{bits}', 'qat', bits)
        del model
    # Uniform joint pruning+quantization baselines: same options as adaptive.
    for bits in BITS:
        for s in SPARSITIES:
            model = attach(copy.deepcopy(base), uniform(base, bits, s))
            train(model, tr, va, a.finetune_epochs*a.rounds, .001, device, freeze_bn=True)
            record(model, f'uniform_q{bits}_s{int(s*100)}', 'uniform_joint', bits, s)
            del model
    # P7: independent target runs, 2-3 re-probe/reallocation/QAT rounds.
    all_probes = list(probes)
    for target in a.targets:
        model, current_probes = None, copy.deepcopy(probes)
        for round_id in range(a.rounds):
            master = copy.deepcopy(base) if model is None else materialize(model)
            if round_id:
                current_probes = probe_candidates(master, probe, device, round_id)
                for r in current_probes: r['target_ratio'] = target
                # Keep separate files, as phase-5 schema intentionally has no target.
                write_csv(a.out/f'probes_target{target}_round{round_id}.csv', current_probes)
            config, detail = allocate(master, current_probes, target, a.tau)
            write_csv(a.out/f'allocation_target{target}_round{round_id}.csv', detail)
            if model is None:
                model = attach(copy.deepcopy(base), config)
            else:
                # Keep FP32 masters, so previously pruned weights can recover if
                # a later allocation restores them. Probe uses effective weights.
                update_config(model, config)
            train(model, tr, va, a.finetune_epochs, .001, device, freeze_bn=True)
            record(model, f'adaptive_{target}x_round{round_id+1}', 'adaptive', bits=0, round_id=round_id+1)
        del model
    plots(rows, probes, a.out/'plots', 'SYNTHETIC SMOKE TEST' if a.smoke else 'ResNet-18 / CIFAR-10')
    for r in rows:
        r['achieved_total_compression'] = rows[0]['size_bytes']/r['size_bytes']
    write_csv(a.out/'results.csv', rows)
    gaps = []
    for b in BITS:
        ptq = next(r for r in rows if r['phase']=='ptq' and r['bits']==b)
        qat = next(r for r in rows if r['phase']=='qat' and r['bits']==b)
        gaps.append({'bits':b, 'ptq_accuracy':ptq['test_accuracy'],
                     'qat_accuracy':qat['test_accuracy'],
                     'qat_minus_ptq_pp':qat['test_accuracy']-ptq['test_accuracy']})
    write_csv(a.out/'ptq_qat_gaps.csv',gaps)
    # Frontier/HV is descriptive final test reporting; selection uses validation.
    baseline_rows = [r for r in rows if r['phase']!='adaptive']
    adaptive_rows = [r for r in rows if r['phase']=='adaptive']
    ref = max(r['size_mb'] for r in rows)*1.05
    summary = {'reference_size_mb': ref, 'accuracy_floor': 0,
               'baseline_hypervolume': hypervolume(baseline_rows, ref),
               'combined_hypervolume': hypervolume(rows, ref),
               'adaptive_hypervolume': hypervolume(adaptive_rows, ref),
               'validation_selected_model': max(rows, key=lambda r:r['validation_accuracy'])['name'],
               'scope': 'Synthetic verification only' if a.smoke else 'Measured CIFAR-10 experiment'}
    summary['hv_gain'] = summary['combined_hypervolume']-summary['baseline_hypervolume']
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2))
    write_csv(a.out/'pareto.csv', pareto(rows))
    # Final Markdown summary uses computed values, never expected slide numbers.
    report = ['# Experiment report', '', summary['scope'], '',
              'Weight-only quantization; dequantized FP32 execution. Structured pruning is filter masking.', '',
              '| Model | Test accuracy (%) | Packed MB |', '|---|---:|---:|']
    report += [f'| {r["name"]} | {r["test_accuracy"]:.2f} | {r["size_mb"]:.5f} |' for r in rows]
    report += ['', f'Hypervolume gain over all uniform baselines: {summary["hv_gain"]:.6f} MB * percentage points.',
               'No guaranteed Pareto dominance is assumed. See README for caveats and report/REPORT.md for methodology.']
    (a.out/'RESULTS_REPORT.md').write_text('\n'.join(report))
    print('Finished. Outputs:', a.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
