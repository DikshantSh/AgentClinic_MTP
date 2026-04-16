#!/usr/bin/env bash
# ============================================================================
# AgentClinic v2.1 — Multi-GPU Watchdog Queue (v2)
# ============================================================================
# Supports two modes:
#
#   MODE 1 — Single command (original behavior):
#     bash gpu_watchdog.sh 15000 60 "bash run_experiments.sh"
#
#   MODE 2 — Queue file (parallel launcher):
#     bash gpu_watchdog.sh --queue gpu_queue.txt 60
#     Reads experiments from gpu_queue.txt, launches each in the background
#     as soon as a GPU has enough free VRAM. Checks continuously.
#
#   MODE 3 — Run a specific experiment from the queue:
#     bash gpu_watchdog.sh --run e2_jsl_med gpu_queue.txt 60
#     Waits for VRAM and launches only the named experiment.
#
# Queue file format (one experiment per line):
#   EXPERIMENT_NAME | VRAM_NEEDED_MIB | COMMAND
#   Lines starting with # are comments, DONE: lines are skipped.
# ============================================================================

set -uo pipefail

EVENT_LOG="logs/watchdog_events.txt"
mkdir -p logs

# ---- Helper: Find a GPU with at least $1 MiB free ----
find_free_gpu() {
    local threshold=$1
    local free_vrams
    free_vrams=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

    local gpu_idx=0
    while read -r vram; do
        if [ "$vram" -ge "$threshold" ] 2>/dev/null; then
            echo "$gpu_idx $vram"
            return 0
        fi
        ((gpu_idx++))
    done <<< "$free_vrams"
    return 1
}

# ---- Helper: Log an event ----
log_event() {
    local name=$1 gpu=$2 vram=$3 cmd=$4
    local ts
    ts=$(date "+%Y-%m-%d %H:%M:%S")
    {
        echo "------------------------------------------------"
        echo "EVENT: LAUNCHING EXPERIMENT"
        echo "Name:         ${name}"
        echo "GPU Index:    ${gpu}"
        echo "VRAM Free:    ${vram} MiB"
        echo "Local Time:   ${ts}"
        echo "Command:      ${cmd}"
        echo "------------------------------------------------"
    } >> "$EVENT_LOG"
}

# ---- Helper: Launch a single experiment in background ----
launch_experiment() {
    local name=$1 gpu=$2 cmd=$3 logdir="logs"

    echo "[$(date)] ✅ Launching '${name}' on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "${logdir}/${name}.log" 2>&1 &
    local pid=$!
    echo "  PID: ${pid} | Log: ${logdir}/${name}.log"
    echo "$pid"
}

# ============================================================================
# MODE 2 & 3: Queue-based execution
# ============================================================================
if [ "${1:-}" = "--queue" ] || [ "${1:-}" = "--run" ]; then
    MODE="$1"
    shift

    # --run mode: extract experiment name filter
    RUN_FILTER=""
    if [ "$MODE" = "--run" ]; then
        RUN_FILTER="$1"
        shift
    fi

    QUEUE_FILE="${1:?ERROR: Queue file path required}"
    INTERVAL="${2:-60}"

    if [ ! -f "$QUEUE_FILE" ]; then
        echo "ERROR: Queue file not found: $QUEUE_FILE"
        exit 1
    fi

    echo "============================================"
    echo "GPU Watchdog Queue v2 — ${MODE} mode"
    echo "Queue File:     ${QUEUE_FILE}"
    echo "Polling Interval: ${INTERVAL}s"
    [ -n "$RUN_FILTER" ] && echo "Filter:         ${RUN_FILTER}"
    echo "============================================"

    # Track which experiments have been launched in this session
    declare -A LAUNCHED

    while true; do
        ALL_DONE=true

        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "$line" ]] && continue
            # Skip completed experiments
            [[ "$line" =~ ^DONE: ]] && continue

            # Parse: NAME | VRAM | COMMAND
            IFS='|' read -r exp_name exp_vram exp_cmd <<< "$line"
            exp_name=$(echo "$exp_name" | xargs)    # trim whitespace
            exp_vram=$(echo "$exp_vram" | xargs)
            exp_cmd=$(echo "$exp_cmd" | xargs)

            # --run filter: skip experiments that don't match
            if [ -n "$RUN_FILTER" ] && [ "$exp_name" != "$RUN_FILTER" ]; then
                continue
            fi

            # Skip if already launched in this session
            if [ "${LAUNCHED[$exp_name]:-}" = "1" ]; then
                continue
            fi

            ALL_DONE=false

            # Check GPU availability for this experiment's VRAM need
            gpu_info=$(find_free_gpu "$exp_vram") || {
                echo "[$(date)] Waiting for ${exp_vram} MiB for '${exp_name}'..."
                continue
            }

            gpu_id=$(echo "$gpu_info" | awk '{print $1}')
            gpu_free=$(echo "$gpu_info" | awk '{print $2}')

            log_event "$exp_name" "$gpu_id" "$gpu_free" "$exp_cmd"
            launch_experiment "$exp_name" "$gpu_id" "$exp_cmd"
            LAUNCHED[$exp_name]=1

            # Wait a moment for the GPU memory to be allocated before checking next
            sleep 10

        done < "$QUEUE_FILE"

        # If all experiments are done or launched, exit
        if $ALL_DONE; then
            echo ""
            echo "[$(date)] ✅ All experiments launched or completed. Watchdog exiting."
            break
        fi

        sleep "$INTERVAL"
    done

    exit 0
fi

# ============================================================================
# MODE 1: Original single-command mode (backwards compatible)
# ============================================================================
THRESHOLD=${1:-15000}
INTERVAL=${2:-60}
shift 2
COMMAND_TO_RUN="$*"

if [ -z "$COMMAND_TO_RUN" ]; then
    echo "ERROR: No command specified."
    echo ""
    echo "Usage:"
    echo "  Single:    bash gpu_watchdog.sh [THRESHOLD] [INTERVAL] \"[COMMAND]\""
    echo "  Queue:     bash gpu_watchdog.sh --queue gpu_queue.txt [INTERVAL]"
    echo "  Specific:  bash gpu_watchdog.sh --run e2_jsl_med gpu_queue.txt [INTERVAL]"
    exit 1
fi

echo "============================================"
echo "GPU Watchdog Queue — Single Command Mode"
echo "Target Threshold: ${THRESHOLD} MiB"
echo "Polling Interval: ${INTERVAL}s"
echo "Command to Run:   ${COMMAND_TO_RUN}"
echo "Monitoring all available GPUs..."
echo "============================================"

while true; do
    gpu_info=$(find_free_gpu "$THRESHOLD") && {
        gpu_id=$(echo "$gpu_info" | awk '{print $1}')
        gpu_free=$(echo "$gpu_info" | awk '{print $2}')

        echo "[$(date)] GPU ${gpu_id} is free (${gpu_free} MiB). Launching command..."
        log_event "single_command" "$gpu_id" "$gpu_free" "$COMMAND_TO_RUN"

        export CUDA_VISIBLE_DEVICES=$gpu_id
        eval "$COMMAND_TO_RUN"
        break
    }

    echo "[$(date)] Still waiting... Next check in ${INTERVAL}s."
    sleep "$INTERVAL"
done
