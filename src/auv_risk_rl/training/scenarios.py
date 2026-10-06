"""复用 train-v1 的确定场景发行，独立标记无障碍任务与验证 split。"""

from dataclasses import replace

from auv_risk_rl.config import ProjectConfig
from auv_risk_rl.env.b0_navigation import B0NavigationEnv
from auv_risk_rl.env.scenario_generator import (
    LocalTrainingScenarioGenerator,
    TrainingScenario,
    TrainingScenarioConfig,
)
from auv_risk_rl.seeding import SeedManager
from auv_risk_rl.training.config import B0HarnessConfig


class B0ScenarioSource:
    """按显式 split/root/index 发行；不依赖策略、风险或终止表现筛选。"""

    def __init__(self, config: B0HarnessConfig, project: ProjectConfig,
                 scenario: TrainingScenarioConfig) -> None:
        """保存冻结分布，构建不消费全局随机流的索引式生成器。"""
        self.config, self.project = config, project
        self.generator = LocalTrainingScenarioGenerator(scenario, project)
        root = config.scenario_root_seed
        self.root_seed = config.training_seed if root is None else root

    def split_seed(self, split: str, purpose: str = 'scenario') -> int:
        """使用既有显式命名空间派生，工程与科研不会共用随机流。"""
        namespace = f'b0/{self.config.run_kind}/{split}/{purpose}'
        return int(SeedManager(self.root_seed).get_rng(namespace).integers(0, 2**63))

    def scenario(self, index: int, split: str = 'train', *,
                 task_profile: str | None = None) -> TrainingScenario:
        """CV 不改变采样律；无障碍另有身份，不改 train-v1 的 1–4 范围。"""
        profile = self.config.task_profile if task_profile is None else task_profile
        if profile not in ('obstacle_free', 'cv_train_v1'):
            raise ValueError('未知场景任务profile。')
        original = self.generator.generate(self.split_seed(split), index)
        if profile == 'cv_train_v1':
            return original
        version = 'B0_OBSTACLE_FREE_V1'
        identity = f'b0-empty-{split}-seed-{original.root_seed}-idx-{index}'
        return replace(original, scenario_id=identity,
                       distribution_version=version, initial_obstacle_states=(),
                       metadata=replace(original.metadata, obstacle_count=0, obstacles=(),
                                        initial_obstacle_overlap_count=0,
                                        scenario_generation_version=version))

    def environment_seed(self, index: int, split: str = 'train') -> int:
        """每个环境独立派生；发行索引恢复后返回相同传感和环境种子。"""
        return int(SeedManager(self.split_seed(split, 'environment')).get_rng(
            f'episode-{index}').integers(0, 2**63))

    def make_env(self, scenario: TrainingScenario) -> B0NavigationEnv:
        """逐值交给 B0，共同世界不重新随机化。"""
        return B0NavigationEnv(self.project, scenario.initial_auv_state,
                               scenario.initial_obstacle_states,
                               scenario.goal_position_ned_m, scenario.scenario_id)
