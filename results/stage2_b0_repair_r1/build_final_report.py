"""从实际确认日志生成精简R1回执和公开样本，不执行环境或学习操作。"""
from __future__ import annotations

import csv
import gzip
import io
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent


def read(path: str) -> Any:
    """读取本轮可信UTF-8结构化证据。"""
    return json.loads((TASK/path).read_text(encoding='utf-8'))


def write(path: str, value: Any) -> None:
    """禁止用非有限值或未知类型伪造JSON。"""
    (TASK/path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def compact_stratum(row: dict[str, Any]) -> dict[str, Any]:
    """公开终点评价分层，不重复完整确认日志元数据。"""
    keys = ('training_seed', 'termination', 'episode_count', 'episode_control_transitions',
            'boundary_subtypes', 'entered_10m_count', 'entered_2m_count', 'reward',
            'reward_components', 'target_distance_and_time')
    return {key: row[key] for key in keys}


def compress_public_episodes() -> dict[str, Any]:
    """保留每episode必要字段，仅压缩公开存储，不删除原CSV。"""
    counts = {}
    for name in ('training_episodes', 'validation_episodes'):
        source = TASK/'public_evidence'/f'{name}.csv'
        destination = source.with_suffix('.csv.gz')
        with source.open('rb') as incoming, destination.open('wb') as outgoing:
            with gzip.GzipFile(filename='', fileobj=outgoing, mode='wb', mtime=0) as compressed:
                while chunk := incoming.read(1024*1024):
                    compressed.write(chunk)
        with source.open(encoding='utf-8', newline='') as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
        counts[name] = dict(rows=rows, raw_bytes=source.stat().st_size,
                            public_gzip_bytes=destination.stat().st_size,
                            raw_file_retained_locally=True)
    # 仅公开原先登记index0的固定100k C/R1轨迹；不是择优，也不改变主终点分母。
    source = TASK/'public_evidence'/'selected_validation_trajectories.csv.gz'
    destination = TASK/'public_evidence'/'fixed_100k_index0_trajectories.csv.gz'
    rows_by_identity: Counter[tuple[str, str]] = Counter()
    with gzip.open(source, 'rt', encoding='utf-8', newline='') as incoming:
        reader = csv.DictReader(incoming)
        with destination.open('wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as outgoing:
                    writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames or [])
                    writer.writeheader()
                    for row in reader:
                        if int(row['at_transition']) == 100000 and int(row['scenario_index']) == 0:
                            writer.writerow(row)
                            rows_by_identity[(row['group'], row['training_seed'])] += 1
    expected = {(group, str(seed)) for group in ('C_3e-4', 'R1_1e-4') for seed in (11, 22, 33)}
    if set(rows_by_identity) != expected:
        raise RuntimeError('固定index0公开轨迹的六个组/seed身份不齐全。')
    counts['trajectory_publication_subset'] = dict(
        selection=('fixed 100k, preregistered index 0, all three C/R1 seed pairs; '
                   'no outcome selection'),
        control_node_records={f'{group}_seed_{seed}': count
                              for (group, seed), count in sorted(rows_by_identity.items())},
        public_gzip_bytes=destination.stat().st_size,
        complete_selected_trajectory_archive_retained_locally=True,
        archive_file='public_evidence/selected_validation_trajectories.csv.gz')
    return counts


def junit_counts(path: str) -> dict[str, int]:
    """直接核对已执行JUnit，不把历史通过计成本轮新结果。"""
    root = ElementTree.parse(TASK/path).getroot()
    cases = root.findall('.//testcase')
    failed = sum(case.find('failure') is not None for case in cases)
    errors = sum(case.find('error') is not None for case in cases)
    skipped = sum(case.find('skipped') is not None for case in cases)
    return dict(passed=len(cases)-failed-errors-skipped, failed=failed, errors=errors,
                skipped=skipped)


def main() -> int:
    """生成结果；任何主计数、身份或工作树差异不符则停止。"""
    analysis = read('analysis/r1_analysis.json')
    audit = read('analysis/final_evidence_audit.json')
    diagnostics = read('analysis/final_fixed_diagnostics.json')
    ledger = read('diag_budget.json')
    state = read('batch_state.json')
    code = analysis['experiment_code_commit']
    protected = ('src', 'tests', 'scripts', 'configs', 'docs/STAGE2_B0_REPAIR_R1.md',
                 'pyproject.toml', 'requirements-b1-tools.txt')
    command = ['git', '-c', f'safe.directory={ROOT.as_posix()}', 'diff', '--name-only', code,
               '--', *protected]
    differences = subprocess.run(command, cwd=ROOT, check=True, capture_output=True,
                                 text=True).stdout.splitlines()
    if differences:
        raise RuntimeError(f'运行后科学代码/配置/测试/登记发生差异：{differences}')
    if (state['status'] != 'COMPLETED' or audit['status'] != 'FINAL_EVIDENCE_AUDIT_PASS'
            or analysis['actual_new_scientific_training_transitions'] != 300000
            or analysis['actual_new_complete_sac_updates'] != 270003):
        raise RuntimeError('实际完成门禁或科研计数不符。')
    final_strata = [compact_stratum(row) for row in audit['termination_strata']
                   if row['group'] == 'R1_1e-4' and row['source'] == 'Val300'
                   and row['scope'] == 'all_registered'
                   and row['transition_point_or_bin_end'] == 100000]
    final_cases = [dict(trajectory_id=row['trajectory_id'], seed=row['training_seed'],
                        failure_type=row['failure_type'], transitions=row['transitions'],
                        minimum_goal_distance_m=row['closest_approach']['distance_m'],
                        minimum_goal_time_s=row['closest_approach']['timestamp_s'],
                        boundary_subtypes=row['boundary_subtypes'],
                        goal_event_consistent=row['goal_event_consistent'],
                        actor_parameters_unchanged=row['actor_parameters_unchanged'])
                   for row in diagnostics['cases']]
    model_summary = [{key: row[key] for key in ('seed', 'transition', 'status', 'bytes',
                                               'complete_sac_updates', 'optimizer_steps', 'alpha')}
                     for row in audit['small_model_audits']]
    resources = dict(
        training_seconds=sum(row['training_seconds'] for row in state['seeds'].values()),
        evaluation_seconds=sum(row['evaluation_seconds'] for row in state['seeds'].values()),
        checkpoint_seconds=sum(row['checkpoint_seconds'] for row in state['seeds'].values()))
    validation = sum(row['evaluation_env_transitions'] for row in state['seeds'].values())
    train_warmup = sum(row['training_warmup_transitions'] for row in state['seeds'].values())
    val_warmup = sum(row['evaluation_warmup_control_transitions']
                     for row in state['seeds'].values())
    entries = ledger['attempts']
    diagnostic_count = len(entries)
    diagnostic_transitions = sum(row['actual_transitions'] for row in entries)
    diagnostic_warmup = sum(row['warmup_control_transitions'] for row in entries)
    if (diagnostic_count > 200 or diagnostic_transitions > 200000
            or any(row['status'] != 'COMPLETED' for row in entries)):
        raise RuntimeError('冻结诊断账本超预算或仍有未完成条目。')
    tests = dict(repository=junit_counts('pytest_all.xml'),
                 phase_a_external=junit_counts('pytest_phase_a.xml'),
                 independent_r1_numerics=junit_counts('math/numerics.xml'))
    if tests['repository'] != dict(passed=621, failed=0, errors=0, skipped=0):
        raise RuntimeError('最终全仓JUnit与真实运行回执不符。')
    publication = compress_public_episodes()
    mechanism = [
        dict(hypothesis='所有boundary都是pitch或终态等于30度表示错误clip',
             level='NOT_SUPPORTED',
             evidence=('V1原18例有8个pitch；选定重放另有位置限制，事件独立定位一致。'
                       'R1全Val300有16 pitch和119位置边界。'),
             limitation='V1全Val300没有子型字段，不能补造全量历史分类。',
             action='不修改边界/终止'),
        dict(hypothesis='实际进入2m目标球后漏判成功', level='NOT_SUPPORTED',
             evidence=('213578个独立0.05s执行分段无首次事件不一致/漏判；'
                       'R1终点26成功均有first2m，874失败均无first2m。'),
             limitation='限登记重放和本轮测得轨迹，不是所有可能状态的证明。',
             action='不放大成功半径'),
        dict(hypothesis='yaw/pitch/执行响应及目标捕获存在多组件控制困难',
             level='SUPPORTED_MECHANISM',
             evidence=('30个冻结反事实分支各从同快照开始，8个接管成功；'
                       '不同案例由yaw或pitch替换改善，部分仍超时或触发其他边界。'),
             limitation='接管成功不是learned策略成功，也不能证明全部失败仅一个原因。',
             action='保留轨迹，不给B0增加控制器'),
        dict(hypothesis='Replay几乎没有成功相关经验', level='NOT_SUPPORTED',
             evidence=('旧首100k成功episode转移占9.932%、5.714%、8.370%；'
                       '成功终点17/12/15不能代表所有成功相关转移。'),
             limitation='当前300k Replay仍不等于历史100k训练时的随机状态。',
             action='不更换Replay/采样律'),
        dict(hypothesis='课程切换导致100k以前全部退化', level='NOT_SUPPORTED',
             evidence='旧monitor30在50k或75k后到100k明显下降，发生于CV切换以前。',
             limitation='后续CV可以进一步影响策略，但不能倒推为此前退化原因。',
             action='本轮不重训CV'),
        dict(hypothesis='降低公共学习率至1e-4恢复固定100k任务完成能力',
             level='NOT_SUPPORTED',
             evidence=('预登记三seed固定100k Val300成功数全部下降，'
                       '跨seed均值6.00%降至2.8889%；九个固定R01–R03仍0成功。'),
             limitation=('只否定本轮具体修复候选的改善，'
                         '不证明学习率在所有预算/配置下无作用。'),
             action='不自动第二候选/加预算'),
        dict(hypothesis='critic误排序或熵动态是后期退化的完整根因',
             level='PLAUSIBLE_HYPOTHESIS',
             evidence='旧alpha下降、Q-loss增加；新较低LR仍未完成简单任务。300k冻结critic对部分动作排序不同。',
             limitation=('V1历史缺50k/75k及100k critic；60条50步随机soft-return尾项未测，'
                         '不能当完整Q真值。'),
             action='建议只用现有中间权重做下轮有界诊断'),
        dict(hypothesis='已确认训练输入/动作/奖励/SAC梯度的L1错误',
             level='NOT_SUPPORTED',
             evidence='18个独立数值测试、六个实际失败输入和213578分段重放在审阅范围一致。',
             limitation='没有确证L1，不等于全域无缺陷证明；诊断工具的roundoff/JSON/时戳记录问题已单独修复。',
             action='不改科学内核，只启用登记的单因素配置'),
    ]
    monitor = []
    for row in analysis['validation_points']:
        stats = row['paired_monitor30']
        monitor.append(dict(group=row['group'], seed=row['training_seed'],
                            at_transition=row['at_transition'], n=30,
                            event_counts=stats['event_counts'], success_rate=stats['success_rate'],
                            mean_reward=stats['moments']['reward']['mean'],
                            subset_at_100k_is_part_of_val300=row['full']))
    result = dict(
        task='STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1',
        generated_at=datetime.now(UTC).isoformat(), batch_status='COMPLETED',
        experiment_code_commit=code, tested_code_commit='ca4e67b88941a19ee1974be8150b58e7a443e5dd',
        tested_to_experiment_protected_git_diff=[], branch='codex/stage2-b0-repair-r1-public',
        historical_control_commit=analysis['control_experiment_commit'],
        branch_decision=read('branch_decision.json'), control_comparability='PASS',
        actual_seed_counts=analysis['seed_counts'],
        scientific_training_steps=analysis['actual_new_scientific_training_transitions'],
        scientific_training_updates=analysis['actual_new_complete_sac_updates'],
        scientific_optimizer_steps=analysis['actual_new_optimizer_steps'],
        reused_c_steps=analysis['reused_control_training_transitions'],
        reused_c_updates=analysis['reused_control_complete_sac_updates'],
        historical_v1_total_steps_not_new=900000,
        validation_env_transitions=validation, training_warmup_controls=train_warmup,
        validation_warmup_controls=val_warmup,
        frozen_diagnostic_trajectories=diagnostic_count,
        frozen_diagnostic_controls=diagnostic_transitions,
        frozen_diagnostic_warmup_controls=diagnostic_warmup,
        failed_scientific_attempts=0, scientific_recomputed_steps=0,
        final_cases=final_cases, endpoint_comparison=analysis['endpoint_comparison'],
        across_training_seed_mean_sample_sd=analysis['across_training_seed_mean_sample_sd'],
        paired_endpoint_changes=analysis['paired_endpoint_changes'], monitor30=monitor,
        final_r1_termination_strata=final_strata, small_models=model_summary,
        actual_completion_gate=audit['completion_gate'], elapsed=resources,
        tests=tests, new_scientific_tests=60, scientific_tests_retired=0,
        original_tolerances_and_ruff_rules_unchanged=True,
        final_suite_computation_observer=read('pytest_all_counts.json'),
        synthetic_accounting_note=(
            '最终全仓实际观测84次完整ordinary更新，其中62次显式合成batch；'
            '不可把嵌套计数相加。另有独立数学检查的Actor步，'
            '早期定向调试未全程部署计数观察器，总合成操作数不冒充已完整测得。'),
        mechanism_evidence=mechanism, reviewed_confirmed_scientific_l1_defects=[],
        scientific_core_modified=[], post_training_protected_git_diff=differences,
        numerical_scientific_failures=0,
        results_tool_failures_preserved=[
            'r1_final_analysis: NumPy event array JSON serialization, '
            'after all nine replays completed; retry reused summaries',
            'r1_final_evidence_audit: separate actual timestamp reads '
            'wrongly required to be equal; ordered UTC check repaired',
            'diagnostic observer goal-root roundoff: retained minimal counterexample, '
            'original world event unchanged',
            'comparability tool mistook split/identity_note publishing annotations '
            'for scenario geometry; raw first failure retained'],
        stage2_scientific_go_established=False, stage2_conditional_go_established=False,
        final_research_hypothesis_no_go_declared=False, stage3_entry='NO-GO / HOLD',
        stage2_decision_reason=('多个seed仍未稳定完成登记简单可达任务；'
                                '降低LR候选没有改善固定终点。第一轮L2已完成，'
                                '有限修复前提尚未耗尽，不能宣布预测/风险/联邦假设失败。'),
        repair_round=1, remaining_l2_or_l3_rounds=1, next_round_authorized=False,
        next_single_suggested_task=('用已保存25k/50k/75k/100k Actor+critic'
                                   '做有界近目标价值/控制诊断并预登记最后一轮单因素R2；'
                                   '须另行授权。'),
        cv_retraining_steps=0, test_id_runs=0, ood_runs=0, federated_work_started=False,
        official_throughput_benchmark_completed=False, publication=publication,
        protocol_word_uploaded=False, model_and_replay_uploaded=False,
        extra_hash_or_zip_sealing_performed=False)
    write('R1_RESULT.json', result)
    lines = [
        '# Stage2 B0 R1 诊断与单因素修复检验结果', '',
        '**批次真实完成；R1学习率候选未改善固定100k任务完成能力。'
        'Stage2 GO未建立，Stage3暂停。**', '',
        f'本机目录 `{ROOT}`。实验代码 `{code}`；冻结登记 `docs/STAGE2_B0_REPAIR_R1.md`，',
        '配置 `configs/stage2_b0_repair_r1.yaml`。被测代码ca4e67b与实际实验9029d60的',
        'src/tests/scripts/configs/登记及依赖Git差异为空；训练后相同保护范围亦无变化。',
        '公开分支codex/stage2-b0-repair-r1-public；实际发布回执见PUBLICATION.json。', '',
        '## 分支选择、可比性和修改', '',
        '先冻结假设/选样规则，完成已有日志、原生公式、独立数学及有界策略重放后选择B。',
        '未在已审阅范围发现确认的科学L1，也无未解决的关键协议冲突；这不是全域无bug证明。',
        '仅公共学习率3e-4→1e-4，原默认与V1仍3e-4。Actor/Q/alpha共用原优化器配置；',
        '从零、空场景、双环境、batch256、Replay500k、starts10000、UTD1及所有其他科学参数保持。',
        'C复用V1前三个100k，不重新跑C，不接续旧模型，不跑CV。初始化网络/targets/alpha、',
        'Actor/Replay/场景/环境派生种子和固定Val几何/潜在噪声逐项相同；',
        '15个验证点的直接场景身份比较全部一致，细节见control_comparability.json。',
        '不同参数导致episode长度变化后，训练场景发放时间线可以不同；不强制假配对。',
        '原SAC、动力学、事件、reward、B0观察、TRAIN_SCENARIO_V1内核未改。',
        '新增/修改仅登记守卫、版本化LR配置、薄单课程运行管理、可撤销只读诊断及相关测试。',
        '诊断observer目标root舍入、结果JSON及回执时戳修复是诊断/记录工具修复，',
        '不改变世界转移或优化数学，不能冒称“发现并修好了V1科学L1”。', '',
        '## Q1—Q4：已确认事实与不足', '',
        '| 问题 | 证据与结论 |', '| --- | --- |',
        '| Q1 首发boundary | 原18固定诊断确为8pitch、10timeout；'
        '终态±30°与最早事件插值一致。所选其他轨迹也有位置包络失败。'
        'R1全900个终点Val有16 pitch_lower和119位置限制，无未分类。'
        'C全Val旧子类NOT_RECORDED，不能补造。 |',
        '| Q2 是否进入2m球 | 独立审计213578个0.05s实际分段无首次事件不一致或漏判；'
        'R1的26成功均有first2m，874失败均无first2m。'
        '固定九例最近距离均大于2m，接近不等于成功。 |',
        '| Q3 更新/执行/评价是否一致 | 六个真实失败234D输入独立误差0；'
        '18项解析/梯度检查通过。动作顺序/Body与NED/执行器响应/'
        '终止mask/真实next_obs/reward一致。Word式53的exp(log_alpha)形式正确，'
        '不能因公开库代理损失不同而替换。 |',
        '| Q4 50k—100k退化为何 | 原monitor在50k或75k后下降，先于课程切换；'
        'alpha下降与Q-loss增加是事实而非完整因果解释。'
        '降低LR后所有seed终点成功数仍下降，具体候选不受支持。'
        '历史中间critic缺失，完整机制仍未建立。 |', '',
        '机制证据等级和反对证据详见R1_RESULT.json。成功episode经验占旧前100k的',
        '9.932%/5.714%/8.370%，不能只数17/12/15个成功终点就称Replay没有成功经验。',
        '六个快照各五条预登记反事实共30条，接管8成功，但不能算learned策略成功；',
        'yaw/pitch替换对不同案例作用不同，支持多组件目标捕获困难，未证明全部只因pitch。',
        '60条300k critic随机50步soft-return诊断未测尾项，不能当完整Q校准真值；',
        'V1历史50k/75k模型及100k critic不存在，不重跑V1伪造它们。',
        'reward四项独立累加及进展首尾抵消核对一致；gamma加权进展另计。',
        '没有额外失败惩罚属于冻结设计；没有改成功半径/姿态界/reward或加入风险过滤。', '',
        '## 固定100k终点：每行Val300，N=300', '',
        '| 组 | seed | 成功 | SR | 边界 | timeout | 平均reward |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in analysis['endpoint_comparison']:
        lines.append(f"| {row['group']} | {row['training_seed']} | {row['goal_success']} | "
                     f"{100*row['success_rate']:.2f}% | {row['operational_boundary_failure']} | "
                     f"{row['task_horizon']} | {row['mean_reward']:.4f} |")
    lines += ['', '跨三个独立训练seed的SR均值±样本SD：C **6.00%±5.13pp**，',
              'R1 **2.89%±3.10pp**。总边界比例46.67%→15.00%，timeout47.33%→82.11%；',
              '不能只取边界减少声称导航或安全改善。三个配对seed的终点SR均下降。',
              'reward四项及分层均值/样本SD见endpoint_comparison.csv和结构化回执。',
              '这是开发Val，不是独立Test；数百episode不是数百训练重复，不生成虚假p值。', '',
              '### R1终点失败细分与接近目标', '',
              '| seed | 首发边界子类 | timeout N | timeout最小目标距离均值 | timeout进入10m / 2m |',
              '| --- | --- | ---: | ---: | ---: |']
    endpoints = {row['training_seed']: row for row in analysis['endpoint_comparison']
                 if row['group'] == 'R1_1e-4'}
    for row in final_strata:
        if row['termination'] != 'task_horizon':
            continue
        seed = row['training_seed']
        distance_mean = row['target_distance_and_time']['minimum_goal_distance_m']['mean']
        lines.append(f"| {seed} | {endpoints[seed]['boundary_subtypes']} | "
                     f"{row['episode_count']} | {distance_mean:.4f} m | "
                     f"{row['entered_10m_count']} / {row['entered_2m_count']} |")
    lines += ['', '进入10m是接近诊断，不是2m成功；其首次进入时间只对真正进入的episode求均值。',
              '细分操作边界和最小目标距离按0.05s已执行分段采集；未执行提议仍留在选定详细轨迹。',
              '无障碍净间距不适用，不以0冒充；训练完整episode与预算未完成片段分开。', '',
              '### 原定monitor30曲线（100k为Val300中的同indices0..29子集）', '',
              '| 组 / seed | step0 | 25k | 50k | 75k | 100k |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for group in ('C_3e-4', 'R1_1e-4'):
        for seed in (11, 22, 33):
            points = sorted([row for row in monitor
                             if row['group'] == group and row['seed'] == seed],
                            key=lambda row: row['at_transition'])
            successes = ' | '.join(str(row['event_counts']['goal_success']) for row in points)
            lines.append(f'| {group} / {seed} | '+successes+' |')
    lines += ['', '每个值为成功数/30；不把监测子集说成Val300，不选择最好25k模型代替终点。',
              '真实曲线见analysis/figures/paired_monitor30_learning_curves.svg、',
              'fixed_100k_val300_raw_seeds.svg和update_diagnostics_comparison.svg；'
              '未平滑或删除失败点。', '',
              '### 固定R01—R03 learned策略诊断', '',
              '| seed | 固定案例 | 原V1 100k结果 | R1 100k结果 | R1最小目标距离 |',
              '| --- | --- | --- | --- | ---: |']
    original = read('frozen_replay_result.json')['cases']
    for row in final_cases:
        suffix = row['trajectory_id'].split(f"r1_final_s{row['seed']}_", 1)[1]
        matches = [case for case in original
                   if case.get('selection') == 'original_fixed'
                   and case['training_seed'] == row['seed']
                   and case['model_transition'] == 100000
                   and suffix in case['trajectory_id']]
        # 匹配依据必须来自真实旧重放身份，缺项不猜测。
        old = (matches[0]['failure_type'] if len(matches) == 1
               else '见原18固定诊断（身份未自动匹配）')
        lines.append(f"| {row['seed']} | {suffix} | {old} | {row['failure_type']} | "
                     f"{row['minimum_goal_distance_m']:.4f} m |")
    lines += ['', 'R1九例0成功、8timeout、1position_D_upper；各一次，不写Replay/不更新参数。',
              '全八维状态/234D观察/Actor分布/动作/提议/事件链的gzip留本机diagnostic_replay/。', '',
              '## 真实计数、执行结束与检查', '',
              '| 项目 | 实测 |', '| --- | ---: |',
              '| 新R1科研训练transition | 300000（11/22/33各100000） |',
              '| 新完整SAC update | 270003（各90001，10000第一次更新） |',
              '| 新optimizer step | 1080012（各360004；不能当完整update） |',
              f'| 独立固定验证transition | {validation} |',
              f'| 训练合法warm-up控制步 | {train_warmup} |',
              f'| 验证合法warm-up控制步 | {val_warmup} |',
              f'| 全部冻结诊断轨迹 / 控制步 / warm-up | {diagnostic_count} / '
              f'{diagnostic_transitions} / {diagnostic_warmup} |',
              '| 失败科研attempt / 科研重算 / 新CV | 0 / 0 / 0 |', '',
              'V1历史900000不是本轮新增；复用C只是其首300000，不与R1拼一条学习曲线。',
              f"实际训练累计{resources['training_seconds']:.3f}s；"
              f"固定验证{resources['evaluation_seconds']:.3f}s；",
              f"保存{resources['checkpoint_seconds']:.3f}s。不是正式throughput测量，也不据此改算法。",
              '实际计算PID37516结束码0，启动器PID38168结束码0分别保存并核对；',
              '结束时间2026-10-07T16:24:49 UTC（本机2026-10-08凌晨），不只报launcher。',
              'batch_state.json与commands原始记录确认3seed COMPLETED，没有仍在运行的训练。', '',
              '本轮最终全仓**621 passed, 0 failed, 0 errors, 0 skipped**；'
              '外置Phase A **13 passed**。',
              '新增科学/调度测试60项，其中独立数值18；未删科学断言、未改容差、无skip/xfail。',
              'Ruff0.6.0与compileall均exit0；最终被测代码ca4e67b，与训练代码保护范围相同。',
              '运行后仅结果/绘图/回执工具有改动，另查其Ruff/compile，不重复已未变的621项。',
              '最终全仓观察器实测84次完整ordinary update，其中62次显式合成batch，',
              '401次Adam和1次SGD；另有单独数学检查Actor步。早期定向调试未全程安装',
              '计数观察器，因此不伪称全轮所有合成操作精确为84；嵌套World/env计数不能相加。',
              '两次结果工具失败（NumPy JSON、分开时戳读取）保留首次stdout/stderr/exit1；',
              '最小结果工具修复后exit0，九条已完成重放从summary复用，没有重训/重复轨迹。',
              '旧docstring等无关风格告警不做全面重构；原throughput blocker仍为历史未解决状态。', '',
              '## 存储、证据与公开边界', '',
              '每seed latest_resume.pt仅保留最近完整点；models/保存25k/50k/75k/100k',
              '小模型（Actor/两个Q/targets/alpha），12份均实际读取核验有限、身份及计数正确。',
              '模型、Replay、完整update JSONL、完整234D细轨迹和Word原件只在本机；',
              '原V1唯一数据/模型/报告未删除或覆盖。没有复制整套项目/venv或生成额外摘要/ZIP。',
              '公开保留全部确认episode的精简CSV.gz、固定index0三对终点轨迹、',
              '终点与分箱CSV、三幅SVG、复算脚本、必要JUnit/结束回执；原未压缩表留本机。',
              '公开轨迹减量只按已登记index0，不按好坏选取；原indices0..2及最早失败全集仍留本机。',
              '原始commands.jsonl留在本机并解除新版本Git跟踪，历史提交保留；',
              '公开COMMANDS.txt保留实际命令、退出码、时间和代码版本，绘图解释器的个人路径明确代称。',
              '实际新增本轮results体积见storage_usage.json（现场字节统计，无摘要）；',
              '不将未能单独归因的Git内部空间或已有venv写成新增数据。',
              '首次过宽公开push被自动审查拒绝后，没有绕过；完整ca4本地提交仍保留，',
              '另建不含ca4祖先的小公开提交9029，已实际push。最终结果同样只上传明确小文件。', '',
              '## 科学判断与停止位置', '',
              '**执行完成不等于科学通过。** 预登记第一轮L2候选未支持改善终点完成能力；',
              '仅空场景复测不能把完整CV/Stage2标为通过。LOCAL §25.3多seed简单任务',
              '及任务效用依据仍不足，GO未建立；CONDITIONAL GO的明确可修复原因尚未确认。',
              '进入Stage3为NO-GO/HOLD；第一轮后有限L2/L3修复尚有一轮，未自动授权。',
              '不提前宣告所有修复耗尽或预测/风险/联邦假设NO-GO。',
              '**唯一下一任务建议：使用已保存的中间Actor/critic完成有界近目标价值/控制诊断，',
              '预登记最后一轮单因素R2，再申请用户授权；不自动训练R2、CV或进入Stage3。**',
              '本轮到此停止。']
    (TASK/'R1_RESULT_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    status_path = ROOT/'docs'/'PROJECT_STATUS.md'
    existing = status_path.read_text(encoding='utf-8')
    historical = existing.split('\n---\n', 1)[1]
    head = '\n'.join([
        '# 当前结果：STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1', '',
        '2026-10-08本机：**R1 BATCH COMPLETED；学习率候选未改善固定100k完成能力。**',
        '科学Stage2 GO未建立，CONDITIONAL GO原因未确认；Stage3进入NO-GO/HOLD。',
        '第一轮有限L2已完成，尚有一轮L2/L3须另行授权，不宣布研究假设被否定。',
        f'实验代码{code}，公开分支codex/stage2-b0-repair-r1-public；实际发布见PUBLICATION.json。',
        '未确认科学L1，仅公共lr3e-4→1e-4；原默认、SAC/动力学/reward/观察/场景均不改。',
        'C复用经真实派生随机流和固定Val核对的V1前三个100k；新11/22/33各100k空场景，',
        '共300000科研transition、270003完整update、1080012optimizer step；无新CV/重算。',
        '实际计算PID37516与启动器38168均exit0，三seed完成，不存在仍在运行的训练worker。',
        '固定100k Val300成功C=35/5/14，R1=19/1/6；SR均值6.00%→2.89%，各seed均下降。',
        'R1终点135边界（16pitch、119位置）、739timeout；固定R01–R03九例0成功。',
        '所选213578实际0.05s分段无首次事件不一致/漏判2m；最后R1失败均未进入2m球。',
        f'本轮冻结诊断{diagnostic_count}条/{diagnostic_transitions}控制步/{diagnostic_warmup}warm-up，未超登记上限。',
        '最终全仓621 passed、外置13 passed，0失败/错误/跳过；Ruff/compileall exit0。',
        '科学/测试/配置/登记与训练代码Git差异为空；后续结果工具修复不修改科学计算。',
        '详细机制等级、计数、曲线、终点和存储见results/stage2_b0_repair_r1/R1_RESULT_REPORT.md、R1_RESULT.json。',
        '原V1及历史验收材料/唯一模型保留，本轮不增加摘要/ZIP封存，不重装环境，不做正式benchmark。',
        '**唯一下一任务建议：利用中间权重做有界近目标价值/控制诊断，预登记最后一轮单因素R2，另行申请授权。**',
        '不自动第二候选、增预算、重跑CV、Stage3、Test/OOD或联邦。', '', '---', historical])
    status_path.write_text(head, encoding='utf-8')
    file_bytes: Counter[str] = Counter()
    file_count = 0
    for path in TASK.rglob('*'):
        if path.is_file():
            relative = path.relative_to(TASK)
            directory = relative.parts[0] if len(relative.parts) > 1 else 'root_evidence'
            file_bytes[directory] += path.stat().st_size
            file_count += 1
    write('storage_usage.json', dict(measured_at=datetime.now(UTC).isoformat(),
                                    measurement=('actual current task-directory file sizes; '
                                                 'no hashes'),
                                    files=file_count, total_bytes=sum(file_bytes.values()),
                                    total_gib=sum(file_bytes.values())/1024**3,
                                    by_top_directory_bytes=dict(sorted(file_bytes.items())),
                                    original_v1_files_deleted_or_overwritten=False,
                                    virtual_environments_copied=0,
                                    git_internal_growth='NOT_MEASURED',
                                    task_code_and_docs_bytes_not_included=True))
    print(json.dumps(dict(status='FINAL_REPORT_WRITTEN_FROM_ACTUAL_RESULTS',
                          code_commit=code, training_transitions=300000, updates=270003,
                          frozen_diagnostics=diagnostic_count,
                          frozen_controls=diagnostic_transitions,
                          publication=publication), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
