# CSE 144 Final Project — Claude Agent Guide

## Project overview
100-class image classification (~1079 train images). RTX 5070 Ti, WSL2, Python 3.12.
Key scripts: `search.py` (hyperparameter sweep), `pipeline.py` (full CV run + Kaggle submission).

## Run queue
`run_queue.md` tracks all runs in three tiers: RUNNING, STAGED, FINISHED.
- **Always read it at session start** to understand current state and what's next.
- **Update it after every run**: move entry from RUNNING → FINISHED (add results + next_actions), promote first STAGED entry to RUNNING (add PID + monitor IDs).
- **Add new entries to STAGED** when the user proposes a new experiment.
- Each entry records: script, models, config flags, PID, monitor task IDs, log file, ETA, crash info, results.
- **Timing**: RUNNING entries have `Started` timestamp. On `pipeline_done`/`search_done`, add `Completed` timestamp and compute `Total time = Completed − Started` (e.g. "4h 12m"). Log both in the FINISHED entry.

## Environment
**Always use Windows Python** for training — native CUDA, no WSL filesystem overhead.
```
Windows Python: /mnt/c/Users/Caleb Cho/code/school/cse144-final/.venv-win/Scripts/python.exe
WSL Python:     /mnt/c/Users/Caleb Cho/code/school/cse144-final/.venv/bin/python3  (do NOT use for training)
```
Working directory: `/mnt/c/Users/Caleb Cho/code/school/cse144-final`

## Running with live monitoring

**IMPORTANT — always use a subagent to launch scripts.** Never run search.py or pipeline.py blocking the main chat. Spawn an Agent, have it start the process, report the PID back, then arm the Monitor in the main chat.

### Standard launch sequence (agent does steps 1-2, main chat does steps 3-4)

**Step 1 — subagent starts the process (Windows Python, nohup to survive terminal close):**
```bash
cd "/mnt/c/Users/Caleb Cho/code/school/cse144-final" && nohup .venv-win/Scripts/python.exe -u search.py > search_run.log 2>&1 &
BGPID=$! && echo "PID=$BGPID" && sleep 5 && head -5 search_run.log
```
Use `nohup` so the process survives WSL terminal close/disconnect (SIGHUP). Without it, closing the terminal silently kills training mid-fold.
Subagent reports the PID back to main chat. User needs the PID to kill the run if needed (multiple runs may be active simultaneously).

**Step 2 — subagent confirms log is live** (wait until search_run.log is non-empty).

**Step 3 — main chat arms the signal Monitor:**
```
Monitor(
  command="tail -n +1 -f search_run.log | grep --line-buffered '##SIGNAL##\\|Traceback\\|RuntimeError\\|OOM\\|CUDA out\\|##DEAD##'",
  persistent=True
)
```

**Step 4 — main chat arms the log staleness watchdog (crash detection):**

Windows Python PIDs are not visible to WSL's `kill -0`, so PID-based watchdogs don't work. Use a log staleness poll instead — fires `##DEAD##` if the log hasn't been updated for 20+ minutes:
```
Monitor(
  command="""while true; do
  sleep 600
  last=$(stat -c %Y "/mnt/c/Users/Caleb Cho/code/school/cse144-final/pipeline_run.log" 2>/dev/null)
  now=$(date +%s)
  age=$(( now - last ))
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 2 2>/dev/null | head -3 | awk '{s+=$1} END {if(NR>0) print int(s/NR); else print 0}')
  if [ $age -gt 1200 ]; then
    echo "##DEAD## log stale ${age}s, GPU=${gpu}% — pipeline likely crashed"
  elif [ "${gpu:-100}" -lt 20 ] && [ $age -gt 600 ]; then
    echo "##DEAD## GPU idle ${gpu}% for 10min+ (log age ${age}s) — pipeline may have crashed"
  fi
done""",
  persistent=True
)
```
Fires `##DEAD##` if: log stale >20 min, OR GPU <20% + log not updated for >10 min. Polls every 10 min. For search_run.log change the log path.
```
The signal Monitor catches `##DEAD##`. Send a PushNotification and check the log tail to determine whether it completed normally or crashed. Note: adjust the log path for search_run.log when monitoring search.

### Killing a run
**Always kill via a subagent** (keeps main chat free). Use SIGKILL — PyTorch ignores SIGTERM mid-training.
Subagent runs:
```bash
kill -9 <PID> && sleep 1 && pgrep -a python | grep search || echo "stopped"
```
Multiple concurrent runs write to different log files (e.g. search_run.log, pipeline_run.log).

### pipeline.py
Same pattern — subagent runs (Windows Python):
```bash
cd "/mnt/c/Users/Caleb Cho/code/school/cse144-final" && nohup .venv-win/Scripts/python.exe -u pipeline.py --export pipeline_result.txt > pipeline_run.log 2>&1 &
BGPID=$! && echo "PID=$BGPID" && sleep 5 && head -5 pipeline_run.log
```
After pipeline completes, read results with:
```bash
cat "/mnt/c/Users/Caleb Cho/code/school/cse144-final/pipeline_result.txt"
```

## ##SIGNAL## protocol
Both scripts emit structured lines: `##SIGNAL## <event> key=val ...`

| Event | Trigger | Push to phone? |
|---|---|---|
| `config_start` | search config begins | no |
| `fold_start` | fold begins | no |
| `fold_done` | fold complete (acc, f1, train_sec) | no |
| `config_done` | config CV complete (acc, f1) | **yes** |
| `error` | config/model failed | **yes** |
| `search_done` | all configs finished | **yes** |
| `model_start` | pipeline model begins | no |
| `model_done` | pipeline model CV complete | **yes** |
| `pipeline_done` | all pipeline models done (best acc) | **yes** |

When monitoring, send a PushNotification for **config_done**, **model_done**, **error**, **search_done**, **pipeline_done**.
Skip fold-level signals (too noisy for phone).

**Always report every signal to main chat** — fold_start, fold_done, config_start, config_done, model_start, model_done, pipeline_done. This keeps the user informed of progress without asking.

**Always append a timestamp to every monitor update message** in the format `(MM/DD H:MMam/pm)`, e.g. `(06/08 4:44pm)`. Get the current time with `date '+%m/%d %-I:%M%P'` via Bash before writing the message. Do NOT embed timestamps in scripts or log output — only in the chat message text.

### PushNotification message formats
- `config_done`: `"[search] cfg {config} {model} — acc={acc} f1={f1}"`
- `model_done`: `"[pipeline] {model} done — acc={acc} f1={f1} ({total_min}m)"`
- `error`: `"[ERROR] cfg {config} {model}: {msg}"`
- `search_done`: `"[search] DONE — {total} configs, results in {results}"`
- `pipeline_done`: `"[pipeline] DONE — best={best} acc={acc}"`

## End-of-run cleanup (do this on every search_done / pipeline_done)

When a run completes, always do ALL of the following before starting the next run:

### 1 — Stop stale monitors
Call `TaskStop` on **both** the signal monitor and the watchdog for the finished run.
- Task IDs are recorded in the RUNNING entry of `run_queue.md` (`Monitors: signal=X, watchdog=Y`).
- Do NOT leave old monitors running — they watch stale log files and fire false ##DEAD## alerts.
- Only the monitors for the *currently active* run should be alive at any time.

### 2 — Update run_queue.md
- Move entry from RUNNING → FINISHED (add Completed timestamp, Total time, results, key findings, next action).
- Promote next STAGED entry to RUNNING (add PID + new monitor task IDs).
- If nothing is launching next, set RUNNING to `_(none)_`.

### 3 — Read results and update report.md
- `search_results.csv` for search runs; `pipeline_result.txt` for pipeline runs.
- Add new section(s) to report.md, update section 5.6 summary table, update section 6 discussion as needed.

### 4 — Discuss and suggest next steps in main chat

### Monitor task ID tracking
Every RUNNING entry in run_queue.md must record:
```
- **Monitors**: signal=<task_id>, watchdog=<task_id>
```
These IDs are used at cleanup time to call TaskStop. Never start a new run without recording both IDs.

## Shorthand commands

### "gpu usage"
Run GPU stats sampled over 10s and report: compute %, VRAM used/total, temp.
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader -l 2 &
GPUPID=$!
sleep 10
kill $GPUPID 2>/dev/null
```
If compute <50% and VRAM headroom >4GB, another training script can likely run in parallel.

### "run queue"
Read `run_queue.md` and reply with a compact summary:
- **RUNNING**: one line per active run — run ID, script, models, current status/progress
- **STAGED**: one line per queued run — run ID, script, key experiment description, priority order
Do not print the full file — just the one-line summaries so the user gets a quick status snapshot.

## Key config
Edit `config.py` to change models/hyperparameters before running pipeline.
Edit `CONFIGS_TO_RUN` set in `search.py` to select which configs to run.

## Kaggle submission
```bash
python3 pipeline.py --submit --message "run description"
```
Competition: `ucsc-cse-144-spring-2026-final-project`

## Current best
DINOv2-base: ~84.24% val accuracy (config 1: unfreeze=2, lr=5e-5, no aug).
DINOv2-Large (facebook/dinov2-large): configs 56-63 in search.py, currently running.
