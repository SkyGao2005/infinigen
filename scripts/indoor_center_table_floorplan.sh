#!/usr/bin/env bash
# Same as indoor_center_table.sh, but uses one of Infinigen's *predefined floor
# plans* (via ``Solver.floor_plan=...``) instead of letting the solver
# generate a random room polygon. This gives you deterministic room
# dimensions so the centre table is always in a known spot, and lets you
# try out Infinigen's "Generate Rooms with Existing Floor Plan" feature.
#
# Usage:
#   bash scripts/indoor_center_table_floorplan.sh [START_SEED] [END_SEED] [OUTPUT_DIR]
#
# Environment variables:
#   ROOM_TYPE   — LivingRoom (default) / DiningRoom / Bedroom / Kitchen
#   FLOOR_PLAN  — JSON file path (recommended) OR dotted python module path.
#                 Default picks the matching JSON plan from
#                 ``infinigen_examples/configs_indoor/floor_plans/``.
#   CONFIGS     — gin configs (default: "singleroom.gin fast_solve.gin")
#   EXTRA       — extra "-p" overrides appended verbatim
#
# Examples:
#   # Default: square LivingRoom with furniture + centre table
#   bash scripts/indoor_center_table_floorplan.sh 0 4 outputs/fp_living
#
#   # Use the dining-room plan (JSON is picked automatically from ROOM_TYPE)
#   ROOM_TYPE=DiningRoom bash scripts/indoor_center_table_floorplan.sh 0 4 outputs/fp_dining
#
#   # Use your own JSON floor plan (centred on origin)
#   FLOOR_PLAN=/path/to/my_plan.json \
#     bash scripts/indoor_center_table_floorplan.sh 0 4 outputs/fp_custom

set -euo pipefail

START_SEED="${1:-${START:-0}}"
END_SEED="${2:-${END:-4}}"
OUTPUT_DIR="${3:-${OUTPUT:-outputs/center_table_floorplan}}"

ROOM_TYPE="${ROOM_TYPE-LivingRoom}"
CONFIGS="${CONFIGS:-singleroom.gin fast_solve.gin}"
EXTRA="${EXTRA:-}"

# Pick a sensible default floor plan (JSON form — sidesteps python import
# issues when the repo is synced to another machine). Run from the infinigen
# repo root so these relative paths resolve.
FP_DIR="infinigen_examples/configs_indoor/floor_plans"
if [[ -z "${FLOOR_PLAN:-}" ]]; then
    case "${ROOM_TYPE}" in
        LivingRoom) FLOOR_PLAN="${FP_DIR}/single_living_room.json" ;;
        DiningRoom) FLOOR_PLAN="${FP_DIR}/single_dining_room.json" ;;
        Bedroom)    FLOOR_PLAN="${FP_DIR}/single_bedroom.json" ;;
        Kitchen)    FLOOR_PLAN="${FP_DIR}/single_kitchen.json" ;;
        *)          FLOOR_PLAN="${FP_DIR}/single_living_room.json" ;;
    esac
fi

if [[ "${FLOOR_PLAN}" == *.json || "${FLOOR_PLAN}" == *.yaml || "${FLOOR_PLAN}" == *.yml ]]; then
    if [[ ! -f "${FLOOR_PLAN}" ]]; then
        echo "[error] floor-plan file not found: ${FLOOR_PLAN}" >&2
        echo "        run this script from the infinigen repo root, or set FLOOR_PLAN to an absolute path." >&2
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}"

ROOM_ARGS=()
if [[ -n "${ROOM_TYPE}" ]]; then
    ROOM_ARGS+=(--room_type "${ROOM_TYPE}")
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(-p ${EXTRA})
fi

for seed in $(seq "${START_SEED}" "${END_SEED}"); do
    scene_dir="${OUTPUT_DIR}/scene_${seed}"
    if [[ -f "${scene_dir}/scene.blend" ]]; then
        echo "[skip] ${scene_dir}/scene.blend already exists"
        continue
    fi
    echo "[generate] seed=${seed} room=${ROOM_TYPE} plan=${FLOOR_PLAN} -> ${scene_dir}"
    # shellcheck disable=SC2086
    python -m infinigen_examples.generate_indoor_center_table \
        --output_folder "${scene_dir}" \
        --seed "${seed}" \
        --task coarse \
        -g ${CONFIGS} \
        --floor_plan "${FLOOR_PLAN}" \
        "${ROOM_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
done

echo
echo "Done. Transfer the scene.blend files to your render box and run:"
echo "  python -m infinigen_examples.generate_indoor_center_table \\"
echo "      --input_folder  ${OUTPUT_DIR}/scene_<SEED> \\"
echo "      --output_folder ${OUTPUT_DIR}/scene_<SEED>/frames \\"
echo "      --seed <SEED> --task render"
