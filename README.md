# LOCAL AUV risk-constrained reinforcement learning

真实已有项目迁入仓库，工程实现截至 PHASE B5.1（TRAIN_SCENARIO_V1）。
工程 Phase、论文中的 B5 Full method、LOCAL 科学 Stage 0—14 是三个不同编号体系。
本轮仅完成 Git 管理及 Stage 0／Stage 1 的非学习验收；不启动训练或 benchmark。

当前结论、证据对应表、实际命令与未解决问题统一见
[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)。历史 B5.1 报告标为历史证据，不能替代本次运行。

科学依据为本机 `handoff/protocol/LOCAL_v2_0.docx`，v2.0，2026-09-17。
该研究原件及私人交接上下文未公开上传；仓库保留代码、版本化配置、独立参考测试和必要小体积结果。
需复核全文时由项目所有者提供原件，不从报告重建协议。

Python >=3.11；依赖见 `pyproject.toml`。RL 声明 PyTorch 2.11.0，Ruff 本轮固定 0.6.0。
复用有效环境，不擅自升级依赖。

```powershell
python -m pytest tests -q
$env:AUV_AUDIT_PROJECT_ROOT=(Get-Location).Path
python -m pytest handoff/reference_tests/test_phase_a_conformance.py -q
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
python scripts/run_stage1_acceptance.py --output-dir results/<new-run>/stage1
```

`scripts/run_stage0_audit.py --output-dir results/<new-run>/audit` 保留旧质量规则，
其代码风格/启发式结论须与 LOCAL 科学 Gate 分开核对，不可把历史结果冒充当前 PASS。
额外摘要封存门禁已退役，版本以 Git 提交、配置和真实测试记录定位。

禁止未经授权启动下一科学 Stage、正式训练、OOD 或联邦实验。
