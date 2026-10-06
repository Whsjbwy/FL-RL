"""仅无损压缩登记轨迹CSV供公开发布；保留本机原文，不计算文件摘要。"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    """压缩后直接流式核对字节；临时产物完成才替换，不涉及科研执行。"""
    root = Path(__file__).resolve().parent
    analysis = json.loads((root / 'analysis/batch_analysis.json').read_text(encoding='utf-8'))
    if analysis['status'] != 'BATCH_DATA_COMPLETE':
        raise ValueError('未完成批次不可发布最终轨迹。')
    source = root / 'public_evidence/selected_validation_trajectories.csv'
    target = source.with_suffix('.csv.gz')
    temporary = target.with_suffix('.gz.partial')
    with source.open('rb') as original, temporary.open('wb') as destination:
        with gzip.GzipFile(filename=source.name, mode='wb', fileobj=destination,
                           compresslevel=6, mtime=0) as compressed:
            shutil.copyfileobj(original, compressed)
    with source.open('rb') as original, gzip.open(temporary, 'rb') as restored:
        while block := original.read(1024 * 1024):
            if restored.read(len(block)) != block:
                raise ValueError('压缩轨迹与原文不一致。')
        if restored.read(1):
            raise ValueError('压缩轨迹包含额外数据。')
    os.replace(temporary, target)
    receipt = dict(
        created_at=datetime.now(UTC).isoformat(),
        experiment_code_commit=analysis['experiment_code_commit'],
        original_local_file=source.name, published_file=target.name,
        original_bytes=source.stat().st_size, compressed_bytes=target.stat().st_size,
        compression='gzip, lossless UTF-8 CSV; no project archive or file digest',
        exact_stream_roundtrip=True, raw_local_file_preserved=True,
        reader='plot_public_evidence.py --prefer-compressed-trajectories',
    )
    (source.parent / 'CSV_STORAGE.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
