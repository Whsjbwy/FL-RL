"""从本批真实完成记录生成最终记账/科学审查报告；无训练、无额外摘要封存。"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def read(path: str) -> Any:
    """读取本轮UTF-8/PowerShell BOM记录，不接受非有限JSON数。"""
    def invalid(value: str) -> None:
        raise ValueError(f'非有限记录: {value}')
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


def junit(path: str) -> dict[str, int]:
    """按实际testcase计数，重复执行的定向测试不与全仓相加。"""
    cases = list(ElementTree.parse(ROOT / path).iter('testcase'))
    errors = sum(case.find('error') is not None for case in cases)
    failures = sum(case.find('failure') is not None for case in cases)
    skipped = sum(case.find('skipped') is not None for case in cases)
    return dict(passed=len(cases)-errors-failures-skipped,
                failed=failures, errors=errors, skipped=skipped)


def main() -> None:
    """仅完成且独立核验的批次可记录BATCH_COMPLETED；不把它改成科学GO。"""
    state = read('batch_state.json')
    analysis = read('analysis/batch_analysis.json')
    diagnostics = read('learned_fixed_diagnostics/learned_fixed_diagnostics.json')
    export = read('public_evidence/EXPORT_RECORD.json')
    assert state['status'] == 'COMPLETED'
    assert analysis['status'] == 'BATCH_DATA_COMPLETE'
    assert read('analysis_status.json')['status'] == 'RESULTS_WRITTEN'
    assert len(analysis['validation_points']) == 42
    assert len(diagnostics['cases']) == 18
    assert all(seed['status'] == 'COMPLETE' for seed in analysis['seed_summaries'])
    assert export['counts']['validation_episodes'] == 2880
    seeds = {int(key): value for key, value in state['seeds'].items()}
    fields = ('transitions', 'updates', 'optimizer_steps', 'evaluation_env_transitions',
              'evaluation_warmup_control_transitions', 'training_warmup_transitions',
              'training_seconds', 'evaluation_seconds', 'checkpoint_seconds')
    totals = {key: sum(value[key] for value in seeds.values()) for key in fields}
    assert totals['transitions'] == state['actual_training_transitions'] == 900000
    assert totals['updates'] == state['actual_sac_updates'] == 870003
    fragments = {'phase_boundary': 0, 'budget_stop': 0, 'external_truncation': 0}
    for seed in analysis['seed_summaries']:
        assert not seed['segment_diagnostics']['excluded_unconfirmed_rows']
        for stats in seed['profiles'].values():
            fragments['phase_boundary'] += stats['phase_boundary_fragments']
            fragments['budget_stop'] += stats['budget_stop_fragments']
            fragments['external_truncation'] += stats['external_truncations']
    boundary_evidence = []
    for case in diagnostics['cases']:
        if case['failure_type'] != 'operational_boundary_failure':
            continue
        trajectory = ROOT / 'learned_fixed_diagnostics' / (
            f"seed_{case['training_seed']}_{case['model_transition']}") / (
                case['case_id']) / 'trajectory.csv'
        with trajectory.open(encoding='utf-8', newline='') as stream:
            last = list(csv.DictReader(stream))[-1]
        boundary_evidence.append(dict(
            seed=case['training_seed'], case_id=case['case_id'],
            final_pitch_rad=float(last['pitch_rad']),
            final_position_ned_m=[float(last[key]) for key in ('north_m', 'east_m', 'down_m')],
            source=trajectory.relative_to(ROOT).as_posix()))
    snapshots = [json.loads(line) for line in (ROOT / 'resource_snapshots.jsonl').read_text(
        encoding='utf-8-sig').splitlines() if line.strip()]
    observer = read('pytest_final_counts.json')
    commands = [json.loads(line) for line in (ROOT / 'commands.jsonl').read_text(
        encoding='utf-8-sig').splitlines() if line.strip()]
    assert all(command['exit_code'] == 0 for command in read('analysis_commands.json'))
    files = [path for path in ROOT.rglob('*') if path.is_file()]
    storage = dict(
        measured_at_utc=datetime.now(UTC).isoformat(),
        task_directory_bytes=sum(path.stat().st_size for path in files),
        local_models_and_resume_bytes=sum(path.stat().st_size for path in files
                                         if path.suffix == '.pt'),
        raw_update_log_bytes=sum(path.stat().st_size for path in files
                                 if path.name == 'update.jsonl'),
        raw_validation_log_bytes=sum(path.stat().st_size for path in files
                                     if path.name == 'validation.jsonl'),
        cleaned_regenerable_fixture_bytes=read('temporary_cleanup.json')['removed_bytes'],
        important_original_data_or_models_deleted=False,
        note='This task new directory measured before final report/commit; Git objects separate.',
    )
    endpoints = [point for point in analysis['validation_points'] if point['full']]
    paired = [point for point in analysis['validation_points']
              if point['at_transition'] in (0, 100000, 300000)]
    result = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(), task='STAGE2_B0_MVP_BATCH_V1',
        batch_status='BATCH_COMPLETED', experiment_code_commit=state['experiment_code_commit'],
        branch='codex/stage2-b0-mvp-v1', repository='https://github.com/Whsjbwy/FL-RL',
        scientific_stage2_gate='DEFERRED_PENDING_BOUNDED_FAILURE_DIAGNOSIS',
        gate_reason='GO evidence not established; CONDITIONAL GO causal diagnosis unconfirmed; '
                    'NO-GO prerequisite of ineffective bounded repairs not yet performed.',
        stage3_authorized=False, scientific_hypothesis_rejected=False,
        per_seed=seeds, actual_totals=totals, fragments=fragments,
        validation_points=42, validation_complete_episodes=2880,
        training_episode_records=export['counts']['training_episodes'],
        complete_training_episodes=sum(value['completed_episodes'] for value in seeds.values()),
        actual_endpoints=endpoints, across_seed_n3=analysis['val300_seed_mean_sd'],
        paired_comparisons=paired,
        learned_diagnostic_events=dict(Counter(case['failure_type']
                                              for case in diagnostics['cases'])),
        learned_diagnostic_env_transitions=diagnostics['diagnostic_env_transitions'],
        learned_diagnostic_warmup_control_transitions=(
            diagnostics['diagnostic_warmup_control_transitions']),
        learned_diagnostic_sac_updates=diagnostics['diagnostic_sac_updates'],
        learned_diagnostic_replay_writes=diagnostics['training_replay_writes'],
        fixed_case_boundary_pitch_evidence=boundary_evidence,
        scientific_failure_or_unplanned_resume_count=0,
        scientific_recomputation_transitions=0,
        numerical_exception_events_observed=0,
        numerical_scope='Completed per-update finite checks and confirmed logs; no global claim '
                        'about unrelated computations.',
        tests=dict(repository=junit('pytest_all.xml'), external_phase_a=junit('pytest_phase_a.xml'),
                   public_export_pure_parser=9, public_plot_pure_parser_latest=11,
                   earlier_entry_setup_failure=junit('pytest_entry.xml'),
                   final_regression_observer=observer['counts'],
                   observer_scope='One final 561-item run only; nested counters not additive; '
                                  'earlier targeted operations not comprehensively instrumented.'),
        timing_scope='training_seconds = harness step wall time minus nested validation. '
                     'Some reset/warmup cost included; warmup counts separate. Initialization, '
                     'planned resume and final analysis not fully included in component sums.',
        warmup_independent_wall_seconds=None,
        elapsed_batch_seconds=(datetime.fromisoformat(state['completed_at'])
                               - datetime.fromisoformat(state['started_at'])).total_seconds(),
        observed_process_peak_rss_bytes=max(s['process_peak_rss_bytes'] for s in snapshots),
        gpu_peak_allocation_not_instrumented=True, storage=storage,
        scientific_source_config_test_drift=False,
        source_integrity_evidence='scientific_git_integrity.stdout.txt; command exit0',
        worker_completed_and_exited=True, worker_stderr_bytes=0,
        launcher_tool_session_observed_exit_code=0,
        worker_exit_code_not_independently_captured=True,
        command_record_count=len(commands),
        original_launcher_record_retained=True,
        no_performance_optimization=True, no_unregistered_training=True,
        no_federated_work=True, no_extra_hash_or_zip_sealing=True,
        next_task='STAGE2_B0_BOUNDED_FAILURE_DIAGNOSIS: pitch-boundary and near-goal timeout; '
                  'no automatic retraining, tuning or Stage3.',
    )
    (ROOT / 'MVP_RESULT.json').write_text(json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = [
        '# STAGE2_B0_MVP_BATCH_V1：真实批次结果与科学审查', '',
        '**BATCH_COMPLETED；Stage2科学判断暂缓（待有界失败诊断），不进入Stage3。**', '',
        '真实项目 `D:\\FL+RL\\AUV_CODEX_HANDOFF_V2_COMPLETE`；本机Python3.13.5、',
        'Torch2.11.0+cu130、RTX5060，复用已验收环境，未重装、升级或更改科学参数。',
        f"实际实验代码 `{state['experiment_code_commit']}`；登记/配置在科研运行前提交并推送。",
        '分支 `codex/stage2-b0-mvp-v1`。结果/绘图工具提交不改变此实验代码身份；',
        '最终公开提交见PUBLICATION.json及远端分支引用，不要求报告包含自身提交ID。', '',
        '## 实际执行与计数', '',
        '| seed | 无障碍transition/update | CV transition/update | '
        '合计transition/update | optimizer steps |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for seed, record in seeds.items():
        lines.append(f"| {seed} | 100000/{record['obstacle_free_updates']} | "
                     f"200000/{record['cv_updates']} | "
                     f"{record['transitions']}/{record['updates']} | "
                     f"{record['optimizer_steps']} |")
    lines.extend([
        '', f"实测总科研训练 **{totals['transitions']} transitions、"
        f"{totals['updates']}完整SAC update、",
        f"{totals['optimizer_steps']} optimizer steps**。"
        '计数与确认后的逐条update、episode日志一致。',
        '同seed保留网络/Adam/alpha/Replay/RNG跨100k课程；其他seed独立初始化。',
        '每seed9999步不更新，10000首次更新；无重复learning_starts；2环境共用预算。',
        f"完整训练episode {result['complete_training_episodes']}，"
        '另6个phase_boundary及6个budget_stop片段；',
        '未完成片段不作事件分母、不补reward、不改Replay终止标记、不额外推进。',
        f"验证 **{totals['evaluation_env_transitions']}** transitions（42点、2880完整episode）；",
        f"训练warm-up {totals['training_warmup_transitions']}、验证warm-up "
        f"{totals['evaluation_warmup_control_transitions']}控制transition，均不写训练Replay、不计学习起步。",
        f"独立learned策略诊断 {diagnostics['diagnostic_env_transitions']} transitions、"
        '90warm-up，0更新/Replay写入。',
        '科研失败/意外中断/重算0；3次100k→CV加载是预登记恢复，不是失败重试。',
        f"实测训练计时 {totals['training_seconds']/3600:.4f}h、验证计时 "
        f"{totals['evaluation_seconds']/3600:.4f}h、checkpoint保存 "
        f"{totals['checkpoint_seconds']:.3f}s；",
        f"批次实际elapsed {result['elapsed_batch_seconds']/3600:.4f}h（不含最后分析）。",
        'warm-up数量独立计账，其wall-clock没有单独instrument：'
        '部分包含在step/reset计时中，',
        '不能把一秒模拟暖机写成一秒实测墙钟。计时用于运行记账，'
        '未开展正式throughput benchmark。', '',
        '## 固定终点Val300（每行分母300，普通B0，不作安全方法比较）', '',
        '| seed | profile/终点 | 成功 | 碰撞 | 操作边界失败 | 超时 | SR | mean reward |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ])
    for point in sorted(endpoints, key=lambda p: (p['profile'], p['training_seed'])):
        stats = point['all_registered_episodes']
        events = stats['event_counts']
        lines.append(f"| {point['training_seed']} | {point['profile']}/{point['at_transition']} | "
                     f"{events['goal_success']} | {events['collision']} | "
                     f"{events['operational_boundary_failure']} | {events['task_horizon']} | "
                     f"{100*stats['success_rate']:.4f}% | "
                     f"{stats['moments']['reward']['mean']:.6f} |")
    lines.extend(['', '跨独立训练seed N=3；样本SD分母n−1，不把episode当独立seed。'])
    for endpoint in analysis['val300_seed_mean_sd']:
        metrics = endpoint['metrics']
        lines.append(f"- {endpoint['profile']} SR **{100*metrics['success_rate']['mean']:.4f}% ± "
                     f"{100*metrics['success_rate']['sample_sd']:.4f}个百分点**；reward "
                     f"{metrics['reward']['mean']:.6f} ± {metrics['reward']['sample_sd']:.6f}。")
    lines.extend([
        '', '## 配对学习前后与全部曲线', '',
        'Val固定root20261006、indices0..299，monitor0..29；三seed相同场景ID、外生CV运动与潜在传感流。',
        '全部点直接训练状态/RNG隔离检查通过；未用Test-ID/OOD/校准，没有滚动换场景。',
        '完整Val300端点的monitor比较仍只取同一0..29子集。', '',
        '| seed | 空场景step0→100k monitor成功数/30 | CV step100k→300k monitor成功数/30 |',
        '| --- | ---: | ---: |',
    ])
    for seed in seeds:
        selected = {(p['profile'], p['at_transition']): p['paired_monitor30']
                    for p in paired if p['training_seed'] == seed}
        counts = [selected[key]['event_counts']['goal_success'] for key in
                  (('obstacle_free', 0), ('obstacle_free', 100000),
                   ('cv_train_v1', 100000), ('cv_train_v1', 300000))]
        lines.append(f'| {seed} | {counts[0]}→{counts[1]} | {counts[2]}→{counts[3]} |')
    lines.extend([
        '', '所有训练/验证点与reward、碰撞/边界/超时、loss/alpha图见analysis/figures，',
        '任务两段分别画图，标100k切换和样本30/300，不平滑、不挑最好25k模型。',
        '图已从公开CSV/gzip重新生成并检查版面；NED Down/俯视/三维坐标一致。',
        '训练路径长度定义为实际控制节点折线；无障碍净间距null/N/A，碰撞0不代表避障能力。', '',
        '## learned策略固定可达性诊断及失败证据', '',
        '上轮有界非学习脚本曾为六例找到可达见证；本轮仅用真实100k/300k模型，每例每seed一次。',
        '**0/18成功，10超时、8操作边界失败**。不是拿旧脚本成功冒充策略学会。',
        '8个边界失败诊断的实际终态pitch均触及±30°，位置仍在操作盒内；具体CSV引用见MVP_RESULT.json。',
        '验证的boundary还包含位置球包络、姿态、速度/角率限制；没有完整终态的验证点不能全部归因为位置越界。',
        'seed11空场景Val index0/1超时时距目标约9.0/8.7m；'
        'seed22 index0边界终止仍距目标3.59m，未进入2m目标球。',
        '固定轨迹indices0/1/2+最早失败index共127条、72864节点，成功与失败均保留。',
        '高正reward/局部进展不能替代到达：冻结reward只有progress、goal、时间和动作平滑项，',
        '无额外失败奖励惩罚；这是冻结语义，不据此擅自改reward或宣布实现bug。', '',
        '## 真实验收、范围及缺陷', '',
        '**本轮全仓561 passed，0 failed/errors/skipped；外置PhaseA13 passed；'
        'Ruff0.6.0/compileall exit0。**',
        '实际测试发生在34f2d4a前驱加工作树，完全相同的被测源码/配置/测试随后提交45f3cc8；',
        '登记与被测代码已在训练前提交推送。历史461+13及Stage0/1记录不当成本轮新运行。',
        '一次定向入口命令因新basetemp父目录不存在，8pass/11setup errors；建立可写目录后19pass，',
        '失败原日志保留，科学断言/容差未改。注册、固定Val隔离、课程边界、'
        '恢复、UTD和日志单测覆盖在561中。',
        '结果导出9项纯解析、压缩绘图最新11项纯解析和其Ruff通过；不与561重复合计。',
        '结果报告helper首次Ruff15处E501、逐项修后余2处、最终0；失败原输出/退出码保留，'
        '没有放宽规则或改变算法。一次日志预览GBK编码错，外层改UTF-8后可读。',
        '最终561项观察器记录84次普通update返回、62次显式batch完整update返回、Adam400/SGD1；',
        '它们有嵌套且只覆盖这次回归，不能相加或冒充本轮全部合成操作。早期定向操作未全量instrument。',
        '训练逐更新有限性检查及全部已确认日志无NaN/Inf异常；未关闭检查、AMP/compile/PER/风险过滤均未新增。',
        '旧39项docstring、3条英文注释/词法告警及throughput blocker仍是历史遗留；未将其改PASS。',
        '轻微状态标签局限：CV恢复初段operation显示INITIALIZING到首个125k点，但真实计数/日志正常前进。',
        '启动记录原样保留；launcher工具会话exit0，worker终码未独立捕获，',
        '但COMPLETED/完整确认日志、两自动分析命令exit0、stderr空和进程退出均实际核对。', '',
        '## 本机保留、公开发布与科学判断', '',
        '本机models/seed_{11,22,33}_{100000,300000}.pt以及seed_*/latest_resume.pt保留，',
        '有效原始episode/update/validation与segment cutoffs保留；'
        '最近全状态可恢复，不存所有25k Replay副本。',
        '公开两个episode CSV及选定轨迹CSV.gz、Val清单、统计/曲线、18固定诊断、必要测试与命令。',
        '压缩21,450,928→7,511,523字节，经直接字节流往返一致检查；原CSV留本机，非项目ZIP/摘要封存。',
        '不上传模型/Replay/虚拟环境/Word原件/私人上下文/大update和rawvalidation日志。',
        f"本轮结果目录当前 {storage['task_directory_bytes']} bytes（"
        f"{storage['task_directory_bytes']/2**30:.3f}GiB，测量范围/时点见JSON）；",
        f"仅清理自生成可再生单测夹具 {storage['cleaned_regenerable_fixture_bytes']} bytes；"
        '原数据/模型未删除。',
        f"已观测进程峰值RAM {result['observed_process_peak_rss_bytes']} bytes；"
        'GPU为资源快照，未instrument分配峰值。',
        '最后normal Git diff确认科研源码、测试、配置、依赖、冻结登记相对45f无漂移。', '',
        '严格按LOCAL§25.3：**GO证据未建立**。固定可达任务0/18、随机CV终点极少成功，',
        '不能只因跑满预算或reward为正晋级。CONDITIONAL GO所需的可学不稳定及具体原因尚未确认；',
        'NO-GO所需的限定修复无效也未执行。因此科学分类**DEFERRED（待有界诊断）**，',
        '不是科学假设被否定，不伪造三类Gate中的通过结论。没有改变预算/奖励/采样律或追加seed。',
        '唯一下一建议：**Stage2 B0姿态限制与接近目标后超时的有界失败诊断**；',
        '固定少数本批失败ID/最终模型，先非学习核对首次终止完整状态、观察归一化、',
        '动作滞后/边界/reward链，再按L1/L2/L3提交具体依据。未经新决定不调参、不重训、不进Stage3。',
    ])
    (ROOT / 'MVP_RESULT_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    status_path = PROJECT / 'docs/PROJECT_STATUS.md'
    old = status_path.read_text(encoding='utf-8')
    if old.startswith('# 当前结果：STAGE2_B0_MVP_BATCH_V1'):
        print(json.dumps(dict(status='FINAL_RECORD_REGENERATED_FROM_SAME_BATCH',
                              project_status_preserved=True, totals=totals), ensure_ascii=False))
        return
    summary = '\n'.join([
        '# 当前结果：STAGE2_B0_MVP_BATCH_V1（2026-10-07本机）', '',
        '**BATCH_COMPLETED；Stage2科学Gate暂缓裁决，GO未建立；Stage3未进入。**',
        '实际实验提交45f3cc87bb79f69ef5257d6bbd5c92bfbea8b898，工作分支codex/stage2-b0-mvp-v1。',
        '三seed11/22/33各100k无障碍+200kCV，均300000真实transition/290001完整update；',
        '总900000/870003，3480012 optimizer steps。原普通SAC/动力学/reward/采样律不变。',
        '42固定验证点和2880完整episode齐全；无障碍Val300成功35/5/14，CV终点2/1/0。',
        '18次learned固定诊断0成功（10超时、8姿态边界失败）；不能当作已学会简单任务。',
        '本轮最终561+13真实通过、Ruff/compileall exit0；旧461+13仍为历史验收。',
        '复用Python3.13.5/Torch2.11.0+cu130/RTX5060；未重装或改变默认参数。',
        '科研有限性检查无异常，源/配置/测试/登记相对45f normal Git diff无变化；结果/工具提交独立。',
        '完整回执、计数、曲线、失败证据与存储见results/stage2_b0_mvp_v1/MVP_RESULT_REPORT.md和MVP_RESULT.json。',
        '本机模型/Replay/原始日志保留；公开轻量CSV/gzip/图及必要测试，发布回执见PUBLICATION.json。',
        '没有额外摘要/ZIP封存、throughput正式benchmark、预算追加、Stage3或联邦工作。',
        'GO证据不足；CONDITIONAL GO原因未确认；NO-GO有限修复前提未完成，不宣告研究假设失败。',
        '**唯一下一任务建议：Stage2 B0姿态边界及近目标超时的有界失败诊断。**',
        '不自动调参、重训或进入下一科学阶段。以下旧状态与“科研0/尚未授权”均为当时历史记录。', '',
        '---', '', '# 历史准备和迁入记录（保留原结论）', '',
    ])
    status_path.write_text(summary+old, encoding='utf-8')
    print(json.dumps(dict(status='FINAL_REVIEW_RECORDED', batch=result['batch_status'],
                          scientific_gate=result['scientific_stage2_gate'], totals=totals,
                          storage=storage), ensure_ascii=False))


if __name__ == '__main__':
    main()
