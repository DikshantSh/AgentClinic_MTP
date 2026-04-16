#!/usr/bin/env bash
# ============================================================================
# AgentClinic v2.1 — Experiment Runner
# ============================================================================
# Runs the full experiment matrix for the M.Tech thesis.
# Usage: bash run_experiments.sh
# Recommended: Run inside screen/tmux for long-running experiments.
# ============================================================================

set -euo pipefail

# --- Configuration ---
SCRIPT="agentic_clinic_v2.py"
SCENARIOS=107
INFERENCES=10
LOG_DIR="logs"
DOCTOR_LLM="HF_mistralai/Mixtral-8x7B-v0.1"
PATIENT_LLM="HF_mistralai/Mixtral-8x7B-v0.1"
MEAS_LLM="HF_mistralai/Mixtral-8x7B-v0.1"
MOD_LLM="HF_mistralai/Mixtral-8x7B-v0.1"

mkdir -p "${LOG_DIR}" results

echo "============================================"
echo "AgentClinic v2.1 — Experiment Suite"
echo "Start time: $(date)"
echo "Scenarios per experiment: ${SCENARIOS}"
echo "Max turns per scenario: ${INFERENCES}"
echo "============================================"

# ============================================================================
# E1: BASELINE — Homogeneous Mistral-7B, No Bias
# ============================================================================
echo ""
echo "[E1] BASELINE: Mistral-7B (homogeneous, no bias)"
echo "  Started at: $(date)"
python "${SCRIPT}" \
    --doctor_llm "${DOCTOR_LLM}" \
    --patient_llm "${PATIENT_LLM}" \
    --measurement_llm "${MEAS_LLM}" \
    --moderator_llm "${MOD_LLM}" \
    --doctor_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
    --patient_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
    --num_scenarios "${SCENARIOS}" \
    --total_inferences "${INFERENCES}" \
    --doctor_bias None \
    2>&1 | tee "${LOG_DIR}/e1_baseline.log"
echo "  [E1] Completed at: $(date)"

# ============================================================================
# E2: HETEROGENEOUS — JSL-Med Doctor, Mistral Patient
# ============================================================================
echo ""
echo "[E2] HETEROGENEOUS: JSL-MedMistral-7B Doctor"
echo "  Started at: $(date)"
python "${SCRIPT}" \
    --doctor_llm "${DOCTOR_LLM}" \
    --patient_llm "${PATIENT_LLM}" \
    --measurement_llm "${MEAS_LLM}" \
    --moderator_llm "${MOD_LLM}" \
    --doctor_model_id "johnsnowlabs/JSL-MedMistral-7B-v2.2" \
    --patient_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
    --num_scenarios "${SCENARIOS}" \
    --total_inferences "${INFERENCES}" \
    --doctor_bias None \
    2>&1 | tee "${LOG_DIR}/e2_jsl_med.log"
echo "  [E2] Completed at: $(date)"

# ============================================================================
# E3: HETEROGENEOUS — Llama-3.1-8B Doctor, Mistral Patient
# ============================================================================
echo ""
echo "[E3] HETEROGENEOUS: Llama-3.1-8B Doctor"
echo "  Started at: $(date)"
python "${SCRIPT}" \
    --doctor_llm "${DOCTOR_LLM}" \
    --patient_llm "${PATIENT_LLM}" \
    --measurement_llm "${MEAS_LLM}" \
    --moderator_llm "${MOD_LLM}" \
    --doctor_model_id "meta-llama/Llama-3.1-8B-Instruct" \
    --patient_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
    --num_scenarios "${SCENARIOS}" \
    --total_inferences "${INFERENCES}" \
    --doctor_bias None \
    2>&1 | tee "${LOG_DIR}/e3_llama3.log"
echo "  [E3] Completed at: $(date)"

# ============================================================================
# E4: BIAS — Recency Bias on Best Doctor Model
# ============================================================================
echo ""
echo "[E4] BIAS: Recency bias (JSL-Med Doctor)"
echo "  Started at: $(date)"
python "${SCRIPT}" \
    --doctor_llm "${DOCTOR_LLM}" \
    --patient_llm "${PATIENT_LLM}" \
    --measurement_llm "${MEAS_LLM}" \
    --moderator_llm "${MOD_LLM}" \
    --doctor_model_id "johnsnowlabs/JSL-MedMistral-7B-v2.2" \
    --patient_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
    --num_scenarios "${SCENARIOS}" \
    --total_inferences "${INFERENCES}" \
    --doctor_bias recency \
    2>&1 | tee "${LOG_DIR}/e4_recency.log"
echo "  [E4] Completed at: $(date)"

# ============================================================================
# E5: BIAS — Confirmation Bias on Best Doctor Model
# ============================================================================
echo ""
echo "[E5] BIAS: Confirmation bias (JSL-Med Doctor)"
echo "  Started at: $(date)"
python "${SCRIPT}" \
    --doctor_llm "${DOCTOR_LLM}" \
    --patient_llm "${PATIENT_LLM}" \
    --measurement_llm "${MEAS_LLM}" \
    --moderator_llm "${MOD_LLM}" \
    --doctor_model_id "johnsnowlabs/JSL-MedMistral-7B-v2.2" \
    --patient_model_id "mistralai/Mistral-7B-Instruct-v0.2" \
    --num_scenarios "${SCENARIOS}" \
    --total_inferences "${INFERENCES}" \
    --doctor_bias confirmation \
    2>&1 | tee "${LOG_DIR}/e5_confirmation.log"
echo "  [E5] Completed at: $(date)"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "End time: $(date)"
echo "Results directory: results/"
echo "Logs directory: ${LOG_DIR}/"
echo "============================================"
echo ""
echo "Results files:"
ls -la results/*.jsonl 2>/dev/null || echo "  (no results files found)"
