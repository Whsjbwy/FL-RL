"""公开CSV绘图适配器的纯合成解析检查；不生成科研图、不执行环境或学习。"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """复用既有日志夹具和导出器；只核对CSV重建，所有数据明确为合成测试。"""
    fixture = load('existing_export_fixture', ROOT / 'development/test_public_export.py')
    adapter = load('public_plot_adapter', ROOT / 'plot_public_evidence.py')
    with tempfile.TemporaryDirectory(prefix='public-plot-synthetic-',
                                     dir=ROOT / 'development') as tmp:
        root = Path(tmp)
        _, analysis = fixture.fixture(root)
        analysis['validation_points'] = []
        for seed in adapter.SEEDS:
            diagnostics = {}
            for kind, row in fixture.MODULE.ANALYSIS.confirmed_records(
                    root, seed, fixture.CODE, diagnostics):
                if kind == 'validation':
                    analysis['validation_points'].append(
                        fixture.MODULE.ANALYSIS.validation_point(row))
        fixture.write(root / 'analysis/batch_analysis.json', analysis)
        fixture.MODULE.export(root)
        public = root / 'public_evidence'
        # 只删除本临时目录刚生成的合成raw日志，证明公开CSV不依赖JSONL原件。
        for path in root.rglob('*.jsonl'):
            path.resolve().relative_to(root.resolve())
            path.unlink()
        reconstructed = adapter.retained_rows(analysis, public)
        assert len(reconstructed) == 42
        assert sum(len(item['episodes']) for item in reconstructed) == 168
        episode = reconstructed[0]['episodes'][0]
        assert episode['initial_position_ned_m'] == [0.0, 0.0, 1.0]
        assert episode['complete'] is True
        assert episode['minimum_clearance_m'] is None
        assert episode['trajectory'][0]['position_ned_m'] == [0.1, 0.0, 1.0]
        assert episode['trajectory'][0]['elapsed_s'] == 0.2
        assert episode['trajectory'][0]['minimum_clearance_m'] is None
        incomplete = deepcopy(analysis)
        incomplete['status'] = 'NOT_COMPLETE'
        fixture.must_fail(lambda: adapter.retained_rows(incomplete, public), 'incomplete analysis')
        source = public / 'validation_episodes.csv'
        original = source.read_text(encoding='utf-8')
        source.write_text(original.replace(fixture.CODE, 'WRONG_CODE', 1), encoding='utf-8')
        fixture.must_fail(lambda: adapter.retained_rows(analysis, public), 'CSV identity')
        source.write_text(original, encoding='utf-8')
        trajectory = public / 'selected_validation_trajectories.csv'
        text = trajectory.read_text(encoding='utf-8')
        compressed = trajectory.with_suffix('.csv.gz')
        with gzip.open(compressed, 'wt', encoding='utf-8', newline='') as stream:
            stream.write(text)
        assert adapter.retained_rows(analysis, public, prefer_compressed=True) == reconstructed
        # gzip优先不能因为本机raw损坏而改读raw；gzip-only公开副本也必须可用。
        trajectory.write_text('BROKEN LOCAL RAW\n', encoding='utf-8')
        assert adapter.retained_rows(analysis, public, prefer_compressed=True) == reconstructed
        trajectory.unlink()
        assert adapter.retained_rows(analysis, public) == reconstructed
        trajectory.write_text(text, encoding='utf-8')
        lines = text.splitlines()
        columns = lines[0].split(',')
        values = lines[1].split(',')
        values[columns.index('north_m')] = '0.5'
        lines[1] = ','.join(values)
        trajectory.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        fixture.must_fail(lambda: adapter.retained_rows(analysis, public), 'changed initial node')
        assert 'torch' not in sys.modules
    print(json.dumps(dict(
        status='PASS', fixture_kind='SYNTHETIC_CSV_PARSER_ONLY',
        checks=['42 fixed points and 168 retained trajectories', 'true initial node',
                'raw float/null/bool preserved', 'no local JSONL required',
                'incomplete analysis refuses', 'CSV identity mismatch refuses',
                'changed initial node refuses', 'gzip/raw reconstruction exactly equal',
                'explicit gzip ignores broken local raw', 'gzip-only copy parses',
                'Torch not imported'],
        environment_transitions=0, sac_updates=0, scientific_figures_written=0)))


if __name__ == '__main__':
    main()
