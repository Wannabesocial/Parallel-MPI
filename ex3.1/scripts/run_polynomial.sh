#!/bin/bash

set -e

EXECUTABLE="./bin/polynomial"
RESULTS_DIR="benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_JSON="${RESULTS_DIR}/mpi_benchmarks_${TIMESTAMP}.json"
MACHINES_FILE="./MPI_instructions/machines"

SIZES=(100000)
PROCS=(1 2 4 8 16 24 32)
ITERATIONS=5

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

[ ! -f "$EXECUTABLE" ] && log_error "Executable not found: $EXECUTABLE (run 'make' first)"
[ ! -f "$MACHINES_FILE" ] && log_error "Machines file not found: $MACHINES_FILE"
command -v mpiexec &> /dev/null || log_error "mpiexec not found"

mkdir -p "$RESULTS_DIR"

echo "Configuration:"
echo "  Sizes: ${SIZES[*]}"
echo "  Processes: ${PROCS[*]}"
echo "  Iterations: $ITERATIONS"
echo "  Output: $OUTPUT_JSON"

extract_time() {
    echo "$1" | grep "$2" | sed -n 's/.*[^0-9]\([0-9][0-9]*\.[0-9]*\) sec.*/\1/p' | head -1
}

MPI_CSV=$(mktemp)

total_configs=$((${#SIZES[@]} * ${#PROCS[@]} * ITERATIONS))
current=0

log_info "Starting MPI benchmark sweep ($total_configs total runs)..."
echo ""

for size in "${SIZES[@]}"; do
    for nproc in "${PROCS[@]}"; do
        for iter in $(seq 1 $ITERATIONS); do
            current=$((current + 1))
            progress=$((current * 100 / total_configs))
            echo -ne "\rProgress: [$progress%] size=$size procs=$nproc iter=$iter     "

            output=$(timeout 600 mpiexec -f "$MACHINES_FILE" -n "$nproc" "$EXECUTABLE" "$size" 2>&1 || true)

            create_time=$(extract_time "$output" "Create polynomials")
            send_time=$(extract_time "$output" "Send data time")
            compute_time=$(extract_time "$output" "Compute time")
            receive_time=$(extract_time "$output" "Receive data time")
            total_time=$(extract_time "$output" "Total execution time")

            echo "$size,$nproc,$iter,$create_time,$send_time,$compute_time,$receive_time,$total_time" >> "$MPI_CSV"
        done
    done
done

echo ""
echo ""
log_success "MPI benchmark sweep completed!"
echo ""

mpi_count=$(wc -l < "$MPI_CSV" | tr -d ' ')
log_info "Collected $mpi_count MPI benchmark entries"

log_info "Calculating baseline times from 1-process runs..."

declare -A BASELINE_SUM
declare -A BASELINE_COUNT

while IFS=, read -r size nproc iter create_time send_time compute_time receive_time total_time; do
    if [ "$nproc" = "1" ] && [ -n "$total_time" ] && [[ "$total_time" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        key="$size"
        if [ -z "${BASELINE_SUM[$key]}" ]; then
            BASELINE_SUM[$key]="$total_time"
            BASELINE_COUNT[$key]=1
        else
            BASELINE_SUM[$key]=$(awk -v a="${BASELINE_SUM[$key]}" -v b="$total_time" 'BEGIN {print a + b}')
            BASELINE_COUNT[$key]=$((BASELINE_COUNT[$key] + 1))
        fi
    fi
done < "$MPI_CSV"

declare -A SERIAL_TIME
for key in "${!BASELINE_SUM[@]}"; do
    SERIAL_TIME[$key]=$(awk -v sum="${BASELINE_SUM[$key]}" -v count="${BASELINE_COUNT[$key]}" 'BEGIN {printf "%.6f", sum / count}')
    echo "  Size $key: baseline (1-proc avg) = ${SERIAL_TIME[$key]} sec"
done

echo ""
log_info "Generating JSON output..."

{
    echo "{"
    echo "  \"metadata\": {"
    echo "    \"timestamp\": \"$TIMESTAMP\","
    echo -n "    \"sizes\": ["
    first=1
    for s in "${SIZES[@]}"; do
        [ $first -eq 0 ] && echo -n ", "
        echo -n "$s"
        first=0
    done
    echo "],"
    echo -n "    \"processes\": ["
    first=1
    for p in "${PROCS[@]}"; do
        [ $first -eq 0 ] && echo -n ", "
        echo -n "$p"
        first=0
    done
    echo "],"
    echo "    \"iterations\": $ITERATIONS,"
    echo "    \"implementation\": \"mpi\","
    echo "    \"machine\": \"$(hostname)\""
    echo "  },"
    echo "  \"results\": ["
    need_comma=0

    while IFS=, read -r size nproc iter create_time send_time compute_time receive_time total_time; do
        [ $need_comma -eq 1 ] && echo ","
        need_comma=1

        serial_time="null"
        speedup="null"
        if [ -n "${SERIAL_TIME[$size]}" ]; then
            serial_time="${SERIAL_TIME[$size]}"
            if [ -n "$total_time" ] && [[ "$total_time" =~ ^[0-9]+\.?[0-9]*$ ]] && [ "$total_time" != "0" ]; then
                speedup=$(awk -v st="$serial_time" -v tt="$total_time" 'BEGIN {printf "%.6f", st / tt}')
            fi
        fi

        [ -z "$create_time" ] && create_time="null"
        [ -z "$send_time" ] && send_time="null"
        [ -z "$compute_time" ] && compute_time="null"
        [ -z "$receive_time" ] && receive_time="null"
        [ -z "$total_time" ] && total_time="null"

        echo -n "    {"
        echo -n "\"size\": $size, "
        echo -n "\"processes\": $nproc, "
        echo -n "\"iteration\": $iter, "
        echo -n "\"create_time\": $create_time, "
        echo -n "\"send_time\": $send_time, "
        echo -n "\"compute_time\": $compute_time, "
        echo -n "\"receive_time\": $receive_time, "
        echo -n "\"parallel_time\": $total_time, "
        echo -n "\"serial_time\": $serial_time, "
        echo -n "\"speedup\": $speedup"
        echo -n "}"
    done < "$MPI_CSV"

    echo ""
    echo "  ]"
    echo "}"
} > "$OUTPUT_JSON"

rm -f "$MPI_CSV"

echo ""
log_success "MPI benchmark completed!"
log_info "Results: $OUTPUT_JSON"
log_info "Total entries: $mpi_count"
