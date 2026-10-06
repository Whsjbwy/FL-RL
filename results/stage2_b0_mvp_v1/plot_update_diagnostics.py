"""只读逐更新原始记录；25k登记区间的loss/alpha诊断，不改变训练或端点选择。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ('actor_loss', 'q1_loss', 'q2_loss', 'alpha_loss', 'alpha')


def main() -> int:
    """使用现有绘图runtime，不给科研环境添加依赖；只读checkpoint确认前缀。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = args.batch_root
    state = json.loads((root / 'batch_state.json').read_text(encoding='utf-8'))
    if state['status'] != 'COMPLETED':
        raise RuntimeError('批次未完成，不能生成最终训练诊断图。')
    output = root / 'analysis/figures'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True, layout='constrained')
    for seed in (11, 22, 33):
        directory = root / f'seed_{seed}'
        bins = {}
        for segment in json.loads((directory / 'segments.json').read_text(encoding='utf-8')):
            cutoff = segment.get('valid_log_sequence')
            if cutoff is None:
                continue
            path = root / segment['path'] / 'update.jsonl'
            if not path.exists():
                continue
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line)
                    if (record.get('run_kind') != 'scientific_training'
                            or record.get('code_version') != state['experiment_code_commit']
                            or record.get('training_seed') != seed
                            or record.get('method') != 'B0_FULL_STATE_ORDINARY_SAC'):
                        raise ValueError('更新记录不是本批同版本B0科研产物。')
                    if record['log_sequence'] > cutoff:
                        continue
                    transition = record['transition']
                    endpoint = ((transition - 1) // 25000 + 1) * 25000
                    initial = dict(n=0, sums={name: 0.0 for name in METRICS},
                                   minimum={}, maximum={})
                    entry = bins.setdefault(endpoint, initial)
                    entry['n'] += 1
                    for name in METRICS:
                        value = record['metrics'][name]
                        entry['sums'][name] += value
                        entry['minimum'][name] = min(entry['minimum'].get(name, value), value)
                        entry['maximum'][name] = max(entry['maximum'].get(name, value), value)
        for endpoint, values in sorted(bins.items()):
            rows.append(dict(training_seed=seed, bin_end_transition=endpoint,
                             profile='obstacle_free' if endpoint <= 100000 else 'cv_train_v1',
                             updates=values['n'], mean={key: value / values['n']
                                                        for key, value in values['sums'].items()},
                             minimum=values['minimum'], maximum=values['maximum']))
        points = [row for row in rows if row['training_seed'] == seed]
        for axis, metric in zip(axes, METRICS, strict=True):
            axis.plot([row['bin_end_transition'] / 1000 for row in points],
                      [row['mean'][metric] for row in points], marker='o', label=f'Seed {seed}')
            axis.set_ylabel(metric)
            axis.axvline(100, color='grey', linestyle=':')
            axis.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel('Global training transitions (thousands); 100k curriculum boundary')
    fig.suptitle('Raw update diagnostics: means in registered 25k intervals\n'
                 'No best-point selection; min/max and raw JSONL retained; not task performance')
    fig.savefig(output / 'update_diagnostics.svg')
    plt.close(fig)
    (root / 'analysis/update_diagnostics.json').write_text(
        json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
