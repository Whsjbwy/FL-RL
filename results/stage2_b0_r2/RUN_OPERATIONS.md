# R2 execution and continuation

This is an operational note, not a scientific result or a changed registration.
The frozen scientific code is `929f26af249cbd070dd4913bdf2eb316ae489d2d`;
later result-only commits do not replace that identity.

The actual serial batch was started on 2026-10-09 with this command:

```powershell
.venv-b1/Scripts/python.exe scripts/run_b0_training.py --r2-registration configs/stage2_b0_r2.yaml --execute --run-kind scientific_training --transition-budget 1800000 --output-dir results/stage2_b0_r2
```

**Do not execute that command again while the current worker is running.**
The live PID and progress are in `batch_state.json`; its PID is the computation
worker, not the Windows virtual-environment redirector. `actual_training_worker_exit.json`
captures that worker's real exit. The recorder also preserves the direct child's
exit in `r2_scientific_batch.worker.json` and the completed command in `commands.jsonl`.
`RUNNING_RECEIPT.json` is only a dated snapshot.

Inspect without starting another process:

```powershell
Get-Content results/stage2_b0_r2/batch_state.json -Raw | ConvertFrom-Json | Select-Object status,current_run,current_operation,actual_training_transitions,actual_sac_updates,last_progress_at
Get-Content results/stage2_b0_r2/actual_training_worker_exit.json
Get-Content results/stage2_b0_r2/r2_scientific_batch.stderr.txt -Tail 20
```

The user explicitly authorized temporary hourly thread follow-up, automation `r2`.
It must stop after this batch's authorized analysis and publication. It may not
retry failures, change the experiment, add budget, or start CV/Stage3.

The following final diagnostics and analysis are **NOT RUN at the time this note
was created**. Run them only after the actual six-run completion and zero worker
exit are confirmed. The diagnostic attempt is exclusive; an existing failed or
completed attempt must not be silently overwritten or repeated.

```powershell
.venv-b1/Scripts/python.exe results/stage2_b0_r2/record_command.py final_fixed_diagnostics -- .venv-b1/Scripts/python.exe results/stage2_b0_r2/final_diagnostics.py --execute --trusted-local-models
.venv-b1/Scripts/python.exe results/stage2_b0_r2/record_command.py final_r2_analysis -- .venv-b1/Scripts/python.exe results/stage2_b0_r2/analyze_r2.py
```

Plot-only mode uses the already installed document runtime's Matplotlib, reads
derived data, and performs no model loading, simulation or learning:

```powershell
& 'C:/Users/MSN/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' results/stage2_b0_r2/analyze_r2.py --plot-only --trajectory-plots
```

Review the actual outputs and LOCAL scientific requirements before assigning a
Stage2 decision. Batch completion, finite losses or engineering checks alone are
insufficient. Keep all raw models, Replay, full input/state traces and original
V1/R1 evidence local. Publish only reviewed small results/code and figures on
`codex/stage2-b0-r2`, without force push or merging main. After final delivery,
stop the temporary follow-up and all automatic long training.

`prepare_publication.py --paths-only` refreshes suggestions without changing
dated pretraining receipts. Review final paths and sizes explicitly; do not
regenerate historical pretraining receipts to make them look like final results.
