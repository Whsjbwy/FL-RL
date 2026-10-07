"""区分科研准备、工程 smoke 与实际运行身份的 B0 配置。"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from auv_risk_rl.rl.config import SACConfig
from auv_risk_rl.seeding import SeedManager


def derived_sac_config(config: SACConfig, *, training_seed: int, run_kind: str) -> SACConfig:
    """显式派生网络初始化/策略/Replay 种子，调用方将有效值记录在配置中。"""
    seeds = SeedManager(training_seed)
    values = {f'{name}_seed': int(seeds.get_rng(f'b0/{run_kind}/rl/{namespace}').integers(
        0, 2**63)) for name, namespace in (('initialization', 'initialization'),
                                         ('actor', 'actor'), ('replay', 'replay'))}
    return replace(config, **values)


@dataclass(frozen=True)
class B0HarnessConfig:
    """显式预算与身份；生产默认值固定，工程仅缩减起步数和 Replay。"""

    run_kind: str = 'preflight'
    task_profile: str = 'obstacle_free'
    training_seed: int = 11
    transition_budget: int | None = None
    num_envs: int = 2
    validation_interval: int = 25000
    validation_episodes: int = 1
    validation_max_steps: int = 1000
    checkpoint_interval: int = 25000
    external_max_steps: int | None = None
    scenario_root_seed: int | None = None
    research_registration: str | None = None
    sac: SACConfig = field(default_factory=SACConfig)
    method: str = 'B0_FULL_STATE_ORDINARY_SAC'

    def __post_init__(self) -> None:
        """拒绝错误方法、隐式预算和工程配置混入生产路径。"""
        if self.method != 'B0_FULL_STATE_ORDINARY_SAC':
            raise ValueError('本入口仅支持 B0 普通 SAC。')
        if self.run_kind not in ('preflight', 'engineering_smoke', 'scientific_training'):
            raise ValueError('必须显式登记合法 run_kind。')
        if self.task_profile not in ('obstacle_free', 'cv_train_v1'):
            raise ValueError('未知 B0 task_profile。')
        integers = (self.num_envs, self.validation_interval, self.validation_episodes,
                    self.validation_max_steps, self.checkpoint_interval)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in integers):
            raise ValueError('调度数量及间隔必须为正整数。')
        if self.validation_max_steps > 1000:
            raise ValueError('独立验证不得超出真实任务时域。')
        for seed in (self.training_seed, self.scenario_root_seed):
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                                     or seed < 0):
                raise ValueError('种子必须是非负整数。')
        if self.transition_budget is not None and (
            isinstance(self.transition_budget, bool)
            or not isinstance(self.transition_budget, int) or self.transition_budget < 1
        ):
            raise ValueError('transition_budget 必须为正整数。')
        if self.run_kind != 'preflight' and self.transition_budget is None:
            raise ValueError('实际采样必须显式声明预算。')
        if self.external_max_steps is not None and (
            isinstance(self.external_max_steps, bool)
            or not isinstance(self.external_max_steps, int) or self.external_max_steps < 1
        ):
            raise ValueError('工程外部截断必须为正整数。')
        actual, default = asdict(self.sac), asdict(SACConfig())
        ignored = {'device', 'initialization_seed', 'actor_seed', 'replay_seed'}
        if self.run_kind == 'engineering_smoke':
            ignored |= {'learning_starts', 'replay_capacity'}
            if self.transition_budget is not None and self.transition_budget > 2048:
                raise ValueError('工程 smoke 单次预算不得超过本轮硬上限。')
            if self.sac.replay_capacity > default['replay_capacity']:
                raise ValueError('工程 Replay 只允许缩减容量。')
            if self.sac.learning_starts > default['learning_starts']:
                raise ValueError('工程 learning_starts 只允许缩减。')
        elif self.external_max_steps is not None:
            raise ValueError('生产配置不得使用工程外部截断。')
        # 仅已登记R1空场景复测允许唯一公共学习率候选；原生产默认值及其他参数不变。
        if (self.run_kind == 'scientific_training'
                and self.research_registration == 'STAGE2_B0_FAILURE_DIAGNOSIS_AND_REPAIR_R1'
                and self.task_profile == 'obstacle_free' and self.training_seed in (11, 22, 33)
                and self.transition_budget == 100000 and self.num_envs == 2
                and self.validation_interval == 25000 and self.checkpoint_interval == 25000
                and self.validation_episodes == 30 and self.validation_max_steps == 1000
                and self.sac.learning_rate == 1e-4):
            ignored.add('learning_rate')
        if any(actual[k] != default[k] for k in actual.keys() - ignored):
            raise ValueError('禁止修改冻结的生产 SAC 参数或网络 batch 规模。')

    def to_dict(self) -> dict[str, Any]:
        """返回 checkpoint 和日志使用的完整身份配置。"""
        return asdict(self)
