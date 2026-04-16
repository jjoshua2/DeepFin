#!/bin/bash
CONFIG=/home/josh/projects/chess/configs/pbt2_small.yaml
BROKER_LOG=/home/josh/projects/chess/runs/pbt2_small/server/shared_broker.out

sed -i "s/distributed_inference_use_compile:.*/distributed_inference_use_compile: false/" "$CONFIG"
sed -i "s/distributed_worker_use_compile:.*/distributed_worker_use_compile: false/" "$CONFIG"

echo "wait_ms | avg_batch | pos/s"
echo "--------|-----------|------"

for WAIT_MS in 0.0 0.5 1.0 2.0 5.0 10.0; do
    sed -i "s/distributed_inference_batch_wait_ms:.*/distributed_inference_batch_wait_ms: $WAIT_MS/" "$CONFIG"

    PYTHONPATH=. python3 -m chess_anti_engine.run --config "$CONFIG" --mode tune > /dev/null 2>&1 &
    PID=$!

    sleep 15  # just enough for workers to connect and start games
    sleep 10  # measure
    AVG=$(tail -3 "$BROKER_LOG" | grep -oP 'avg [\d.]+' | awk '{sum+=$2; n++} END {if(n>0) printf "%.0f", sum/n; else print "?"}')
    PPS=$(tail -3 "$BROKER_LOG" | grep -oP '[\d.]+ pos/s' | awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print "?"}')

    printf "%5s   | %6s    | %s\n" "$WAIT_MS" "$AVG" "$PPS"

    kill $PID 2>/dev/null
    wait $PID 2>/dev/null
    ray stop --force > /dev/null 2>&1
    sleep 2
done

sed -i "s/distributed_inference_use_compile:.*/distributed_inference_use_compile: true/" "$CONFIG"
sed -i "s/distributed_worker_use_compile:.*/distributed_worker_use_compile: true/" "$CONFIG"
echo "Done. Compile re-enabled."
