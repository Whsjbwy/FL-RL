"""单进程执行已登记B0三seed固定课程；批次调度不改普通SAC更新数学。"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_risk_rl.config import load_project_config
from auv_risk_rl.env.scenario_generator import load_training_scenario_config
from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.rl.replay import FIELDS
from auv_risk_rl.training.config import B0HarnessConfig, derived_sac_config
from auv_risk_rl.training.mvp_registration import REGISTRATION_ID, git, verify_code_identity


def utc() -> str:
    """记录实际UTC，不以预计时间伪造运行完成。"""
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    """小状态文件完整写入后替换，错误不能损坏上份有效状态。"""
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    """Windows只读查询PID，避免向用户进程发送信号或终止请求。"""
    if sys.platform != 'win32':
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return False
        raise OSError(error, '不能确认旧PID状态，拒绝删除锁。')
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError('不能读取旧PID退出状态。')
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


class BatchLock:
    """互斥锁记录真实PID；过期锁只有显式resume且旧进程不存在才移除。"""

    def __init__(self, path: Path, *, resume: bool) -> None:
        """原子创建锁，不能凭一次工具超时启动第二个训练进程。"""
        self.path = path
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if not process_alive(int(old['pid'])):
                if not resume:
                    raise RuntimeError('发现过期锁，须显式resume审查旧运行。')
                path.unlink()
            else:
                raise RuntimeError(f"已有批次进程PID={old['pid']}，拒绝重复启动。")
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'created_at': utc()}, stream)

    def close(self) -> None:
        """只释放本进程拥有的锁。"""
        if self.path.exists():
            identity = json.loads(self.path.read_text(encoding='utf-8'))
            if identity.get('pid') == os.getpid():
                self.path.unlink()


class SegmentLog:
    """缓冲保留原始episode/update，恢复开新段并明确旧尾部的有效序号。"""

    def __init__(self, output: Path, seed: int, *, parent: Path | None,
                 cutoff: int | None, registration_id: str = REGISTRATION_ID) -> None:
        """不覆盖旧日志；checkpoint之后未确认的旧尾部保留而不重复合并。"""
        self.output, self.seed = output, seed
        self.registration_id = registration_id
        self.seed_dir = output / f'seed_{seed}'
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.seed_dir / 'segments.json'
        self.metadata = (json.loads(self.metadata_path.read_text(encoding='utf-8'))
                         if self.metadata_path.exists() else [])
        if self.metadata and parent is not None:
            self.metadata[-1].update(valid_log_sequence=cutoff, status='PAUSED')
        self.segment_id = f'segment_{len(self.metadata) + 1:04d}'
        self.directory = self.seed_dir / 'segments' / self.segment_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata.append(dict(segment_id=self.segment_id,
                                  path=self.directory.relative_to(output).as_posix(),
                                  parent_checkpoint=str(parent) if parent else None,
                                  valid_log_sequence=cutoff, status='RUNNING', created_at=utc()))
        atomic_json(self.metadata_path, self.metadata)
        self.streams: dict[str, Any] = {}
        self.writes = 0

    def __call__(self, kind: str, record: dict[str, Any]) -> None:
        """训练不保留每个失败episode的巨大轨迹；验证固定轨迹另存，原始指标完整保留。"""
        if kind == 'episode':
            record.pop('trajectory', None)
        record.update(training_seed=self.seed, segment_id=self.segment_id,
                      registration_id=self.registration_id)
        if kind not in self.streams:
            self.streams[kind] = (self.directory / f'{kind}.jsonl').open(
                'a', encoding='utf-8', buffering=1024 * 1024)
        self.streams[kind].write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.writes += 1
        if self.writes % 100 == 0:
            self.flush(sync=False)

    def flush(self, *, sync: bool) -> None:
        """checkpoint前落盘，常规每100条flush，不关闭逐update有限性检查。"""
        for stream in self.streams.values():
            stream.flush()
            if sync:
                os.fsync(stream.fileno())

    def confirm(self, sequence: int, status: str) -> None:
        """记录已安全checkpoint确认的有效前缀，保留失败尾部用于故障审查。"""
        self.flush(sync=True)
        self.metadata[-1].update(valid_log_sequence=sequence, status=status, updated_at=utc())
        atomic_json(self.metadata_path, self.metadata)

    def close(self) -> None:
        """关闭全部缓冲日志。"""
        for stream in self.streams.values():
            stream.close()


def resource_identity(output: Path) -> dict[str, Any]:
    """按真实Replay schema估算存储并读取本机CPU/RAM/显卡，不运行性能benchmark。"""
    row_bytes = sum(int(np.prod(shape or (1,))) * np.dtype(dtype).itemsize
                    for shape, dtype in FIELDS.values())
    ram: dict[str, Any] = {'total_bytes': None, 'available_bytes': None}
    if sys.platform == 'win32':
        class MemoryStatus(ctypes.Structure):
            """Windows GlobalMemoryStatusEx的原生只读结构。"""
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ('total', 'available', 'total_page', 'available_page',
                           'total_virtual', 'available_virtual', 'extended')]]
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            ram = {'total_bytes': memory.total, 'available_bytes': memory.available}
    fields = '--query-gpu=name,driver_version,memory.total,memory.used'
    query = subprocess.run(['nvidia-smi', fields,
                            '--format=csv,noheader'], capture_output=True, text=True, check=False)
    record = dict(created_at=utc(), interpreter=sys.executable, python=sys.version,
                  torch=str(torch.__version__), torch_file=torch.__file__, cuda=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), cpu_logical_cores=os.cpu_count(),
                  torch_num_threads=torch.get_num_threads(), ram=ram,
                  gpu=query.stdout.strip(), nvidia_smi_exit_code=query.returncode,
                  free_disk_bytes=shutil.disk_usage(output).free,
                  replay_bytes_per_transition=row_bytes,
                  replay500k_raw_bytes=500000 * row_bytes,
                  replay300k_chunk_bytes=74 * 4096 * row_bytes,
                  log_budget_bytes_estimate=1024**3,
                  storage_note='ESTIMATE: three latest 300k resumes plus one atomic temporary; '
                               'small actors and <=1GiB logs; no exact runtime forecast')
    atomic_json(output / 'environment_resources.json', record)
    if record['free_disk_bytes'] < 8 * 1024**3:
        raise OSError('批次checkpoint及临时替换至少预留8GiB磁盘。')
    if ram['available_bytes'] is not None and ram['available_bytes'] < 5 * 1024**3:
        raise MemoryError('完整恢复/验证快照和Replay至少预留5GiB可用RAM。')
    if str(torch.__version__) != '2.11.0+cu130' or not torch.cuda.is_available():
        raise RuntimeError('登记的已验收Torch2.11.0+cu130/CUDA目标不可用，拒绝静默CPU回退。')
    test = torch.ones(2, device='cuda') * 2
    torch.cuda.synchronize()
    if not bool(torch.isfinite(test).all()):
        raise FloatingPointError('CUDA入口张量检查非有限。')
    record['gpu_name'] = torch.cuda.get_device_name()
    record['device_capability'] = list(torch.cuda.get_device_capability())
    record['cuda_tensor_check'] = 'PASS'
    atomic_json(output / 'environment_resources.json', record)
    return record


def harness_config(registration: dict[str, Any], seed: int) -> B0HarnessConfig:
    """科研配置严格保持冻结规模；独立seed从全新普通Agent开始。"""
    if seed not in registration['training_seeds']:
        raise ValueError('seed不在本批批准列表。')
    sac = derived_sac_config(SACConfig(**registration['sac'], device='cuda'),
                            training_seed=seed, run_kind='scientific_training')
    return B0HarnessConfig(run_kind='scientific_training', task_profile='obstacle_free',
                           training_seed=seed, transition_budget=300000, num_envs=2,
                           validation_episodes=30, research_registration=REGISTRATION_ID, sac=sac)


def save_actor(harness: Any, output: Path, seed: int) -> None:
    """保留固定终点模型，不复制Replay，不按best曲线选择。"""
    models = output / 'models'
    models.mkdir(exist_ok=True)
    path = models / f'seed_{seed}_{harness.transitions}.pt'
    value = dict(format='b0-mvp-actor-v1',
                 actor={key: tensor.detach().cpu() for key, tensor
                        in harness.agent.actor.state_dict().items()},
                 sac_config=asdict(harness.agent.config), code_version=harness.code_version,
                 seed=seed, transition=harness.transitions, run_kind='scientific_training',
                 method=harness.config.method, registration_id=REGISTRATION_ID)
    temporary = path.with_suffix('.pt.partial')
    with temporary.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def finalize_results(root: Path, state: dict[str, Any]) -> None:
    """整批实际完成后生成原始统计、一次固定策略诊断与图；不自动判科研GO。"""
    if state.get('status') != 'COMPLETED':
        raise RuntimeError('未完成批次不能生成最终学习结果回执。')
    verify_code_identity(root, state['experiment_code_commit'])
    output = root / state['registration']['output_directory']
    commands = [
        [sys.executable, str(root / 'scripts/analyze_b0_mvp.py'),
         '--batch-root', str(output), '--learned-diagnostics', '--trusted-local-models'],
        [str(Path('C:/Users/MSN/.cache/codex-runtimes/codex-primary-runtime/dependencies')
             / 'python/python.exe'), str(root / 'scripts/analyze_b0_mvp.py'),
         '--plot-only', '--analysis-json', str(output / 'analysis/batch_analysis.json'),
         '--output-dir', str(output / 'analysis/figures')],
    ]
    receipt = []
    for index, command in enumerate(commands):
        start = utc()
        stdout_path = output / f'final_analysis_{index}.stdout.txt'
        stderr_path = output / f'final_analysis_{index}.stderr.txt'
        with stdout_path.open('a', encoding='utf-8') as stdout, (
            stderr_path.open('a', encoding='utf-8')
        ) as stderr:
            process = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, check=False)
        receipt.append(dict(command=command, start_utc=start, end_utc=utc(),
                            exit_code=process.returncode,
                            experiment_code_commit=state['experiment_code_commit']))
        atomic_json(output / 'analysis_commands.json', receipt)
        if process.returncode:
            atomic_json(output / 'analysis_status.json', dict(
                status='ANALYSIS_FAILED', failed_command=command,
                note='Batch results remain unchanged; no automatic retraining.'))
            raise RuntimeError('批次训练完成，结果分析命令失败；保留真实错误，不自动重新训练。')
    atomic_json(output / 'analysis_status.json', dict(
        status='RESULTS_WRITTEN', completed_at=utc(),
        stage2_scientific_decision='AWAITING_RAW_EVIDENCE_REVIEW',
        stage3_started=False, scientific_training_beyond_registration=0,
        github_results_status='AWAITING_PUBLICATION_REVIEW'))


def run_batch(root: Path, registration: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    """在本机连续执行六个固定课程job；正常结果差不改变seed、顺序或预算。"""
    from auv_risk_rl.training.curriculum import B0CurriculumHarness
    from auv_risk_rl.training.fixed_validation import FixedValidationPool

    output = root / registration['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'batch_state.json'
    if state_path.exists() and not resume:
        raise RuntimeError('已有批次状态；拒绝重复从零训练，请显式resume。')
    if resume and not state_path.exists():
        raise RuntimeError('没有批次状态可恢复。')
    if not resume:
        old_directories = [output / f'seed_{seed}' for seed in (11, 22, 33)]
        old_directories.append(output / 'models')
        if any(path.exists() and any(path.iterdir()) for path in old_directories):
            raise RuntimeError('新批次输出已有孤立模型/seed日志或checkpoint，拒绝拼入旧产物。')
    state = (json.loads(state_path.read_text(encoding='utf-8')) if resume else dict(
        registration_id=REGISTRATION_ID, registration=registration,
        experiment_code_commit=git(root, 'rev-parse', 'HEAD'), created_at=utc(),
        status='PENDING', completed_jobs=[], seeds={}))
    if state['registration'] != registration or state['registration_id'] != REGISTRATION_ID:
        raise ValueError('批次恢复登记或日程不符。')
    if state['status'] == 'FAILED':
        raise RuntimeError('失败批次不得自动重试；须先审查真实故障及受影响预算。')
    if state['status'] == 'COMPLETED':
        return state
    verify_code_identity(root, state['experiment_code_commit'])
    lock = BatchLock(output / 'batch.lock', resume=resume)
    logger: SegmentLog | None = None
    active_harness: Any = None
    step_timer: float | None = None
    try:
        resource_identity(output)
        project = load_project_config(root / registration['project_config'])
        scenario = load_training_scenario_config(root / registration['scenario_config'])
        pool = FixedValidationPool(project, scenario, root_seed=20261006)
        atomic_json(output / 'validation_manifest.json', pool.compact_manifest())
        state.update(status='RUNNING', pid=os.getpid(), started_at=utc())
        atomic_json(state_path, state)
        for job in registration['execution_order']:
            if job in state['completed_jobs']:
                continue
            seed, stop = job['seed'], job['stop']
            seed_dir = output / f'seed_{seed}'
            seed_dir.mkdir(exist_ok=True)
            checkpoint = seed_dir / 'latest_resume.pt'
            row = state['seeds'].setdefault(str(seed), dict(training_seconds=0.0,
                                                          evaluation_seconds=0.0,
                                                          checkpoint_seconds=0.0))
            row.update(status='RUNNING')
            state.update(current_seed=seed, current_stop=stop, current_operation='INITIALIZING')

            def evaluate(current: Any, profile: str, full: bool,
                         row: dict[str, Any] = row) -> dict[str, Any]:
                """固定池与主训练计时分开；状态文件展示真实评估进度。"""
                state.update(current_operation='EVALUATING', evaluation_profile=profile,
                             evaluation_at_transition=current.transitions, evaluation_full=full,
                             last_progress_at=utc())
                atomic_json(state_path, state)
                start = time.perf_counter()

                def on_episode(index: int, count: int, steps: int, warmup: int) -> None:
                    """每个实际验证案例完成后更新进展，不凭预期步数伪造计数。"""
                    state.update(current_validation_index=index,
                                 current_validation_completed_episodes=count,
                                 current_validation_transitions=steps,
                                 current_validation_warmup_transitions=warmup,
                                 last_progress_at=utc())
                    atomic_json(state_path, state)

                try:
                    result = pool.evaluate(current, profile, full=full,
                                           progress_callback=on_episode)
                finally:
                    row['evaluation_seconds'] += time.perf_counter() - start
                state['current_operation'] = 'TRAINING'
                return result

            harness = B0CurriculumHarness(harness_config(registration, seed), project, scenario,
                                          code_version=state['experiment_code_commit'],
                                          evaluation_callback=evaluate)
            active_harness = harness
            if checkpoint.exists():
                harness.load_checkpoint(checkpoint, trusted_local=True)
            elif stop != 100000:
                raise RuntimeError('CV阶段缺少本seed100k完整状态，不能用另一seed或新初始化。')
            logger = SegmentLog(output, seed, parent=checkpoint if checkpoint.exists() else None,
                                cutoff=harness.log_sequence if checkpoint.exists() else None)
            harness.log_sink = logger

            def progress(*, harness: Any = harness, row: dict[str, Any] = row,
                         seed: int = seed) -> None:
                """累计数为真实harness/Agent计数，不把计划预算填为实测。"""
                row.update(transitions=harness.transitions,
                           updates=harness.agent.counters['gradient_updates'],
                           optimizer_steps=sum(harness.agent.counters[key]
                                               for key in ('actor', 'q1', 'q2', 'alpha')),
                           profile=harness.current_profile, replay_size=len(harness.agent.replay),
                           evaluation_env_transitions=harness.evaluation_env_transitions,
                           evaluation_warmup_control_transitions=(
                               harness.evaluation_warmup_control_transitions),
                           training_warmup_transitions=harness.started_episodes * 5,
                           completed_episodes=harness.completed_episodes,
                           phase_transition_counts=harness.phase_transition_counts,
                           last_progress_at=utc())
                done = sum(item.get('transitions', 0) for item in state['seeds'].values())
                seconds = sum(item['training_seconds'] for item in state['seeds'].values())
                state.update(actual_training_transitions=done,
                             actual_sac_updates=sum(item.get('updates', 0)
                                                    for item in state['seeds'].values()),
                             last_progress_at=utc(), remaining_training_hours_estimate=(
                                 (900000 - done) * seconds / done / 3600 if done > 0 else None),
                             estimate_excludes_future_evaluation_and_save=True)
                atomic_json(state_path, state)
                print(json.dumps(dict(seed=seed, transition=harness.transitions,
                                      updates=row['updates'], profile=row['profile'],
                                      operation=state['current_operation'], time=utc())),
                      flush=True)

            def save(status: str = 'RUNNING', *, harness: Any = harness,
                     logger: SegmentLog = logger, checkpoint: Path = checkpoint,
                     row: dict[str, Any] = row, progress: Any = progress) -> None:
                """只在完整边界安全写最新恢复点；先确认原始日志已经落盘。"""
                logger.flush(sync=True)
                state['current_operation'] = 'CHECKPOINT_SAVE'
                atomic_json(state_path, state)
                start = time.perf_counter()
                harness.save_checkpoint(checkpoint)
                row['checkpoint_seconds'] += time.perf_counter() - start
                logger.confirm(harness.log_sequence, status)
                row['checkpoint'] = str(checkpoint)
                row['checkpoint_transition'] = harness.transitions
                state['current_operation'] = 'TRAINING'
                progress()

            if harness.transitions == 0:
                harness.ensure_validation('obstacle_free', full=False)
                save()
            if stop == 300000 and harness.current_profile != 'cv_train_v1':
                raise RuntimeError('100k课程切换未完成，拒绝隐式放宽恢复身份。')
            while harness.transitions < stop:
                before_eval = row['evaluation_seconds']
                start = step_timer = time.perf_counter()
                result = harness.step()
                row['training_seconds'] += max(0.0, time.perf_counter() - start
                                                - (row['evaluation_seconds'] - before_eval))
                step_timer = None
                if result['checkpoint_due'] and harness.transitions < stop:
                    save()
                elif harness.transitions % 100 == 0:
                    progress()
            save_actor(harness, output, seed)
            if stop == 100000:
                row['obstacle_free_updates'] = harness.agent.counters['gradient_updates']
                harness.switch_to_cv()
                harness.ensure_validation('cv_train_v1', full=False)
                row['status'] = 'PAUSED'
            else:
                row['cv_updates'] = (harness.agent.counters['gradient_updates']
                                     - row['obstacle_free_updates'])
                harness.record_final_budget_stop()
                row['status'] = 'COMPLETED'
            save(row['status'])
            state['completed_jobs'].append(job)
            progress()
            logger.close()
            logger = None
            del progress, save, evaluate
            active_harness = None
            del harness
            torch.cuda.empty_cache()
        state.update(status='COMPLETED', completed_at=utc(), current_operation='BATCH_FINISHED',
                     scientific_stage2_decision='AWAITING_RESULT_ANALYSIS')
        atomic_json(state_path, state)
        return state
    except BaseException:
        state.update(status='FAILED', failed_at=utc(), traceback=traceback.format_exc(),
                     recovery_note='No checkpoint from a failed partial transition/update. '
                                   'Review failure and last valid resume before any new attempt.')
        if active_harness is not None:
            state['failure_actual_counters'] = active_harness.summary()
            metadata = active_harness.failure_metadata or {}
            state['failure_uncommitted_evaluation_env_transitions'] = metadata.get(
                'evaluation_env_transitions')
            state['failure_uncommitted_warmup_transitions'] = metadata.get(
                'warmup_control_transitions')
        if step_timer is not None:
            state['failure_step_wall_seconds_including_nested_evaluation'] = (
                time.perf_counter() - step_timer)
        if logger is not None:
            logger.flush(sync=True)
            logger.metadata[-1]['status'] = 'FAILED'
            atomic_json(logger.metadata_path, logger.metadata)
        atomic_json(state_path, state)
        raise
    finally:
        if logger is not None:
            logger.close()
        lock.close()
