#!/bin/bash
# Start the ISL demo server in the background (logs: server.log, pid: server.pid).
cd "$(dirname "$0")"
PORT="${1:-8080}"
PYTHON="${PYTHON:-python3}"   # e.g. PYTHON=~/miniconda3/envs/isl_demo/bin/python ./start.sh
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
    echo "Already running (pid $(cat server.pid)). Use ./stop.sh first."
    exit 1
fi
nohup "$PYTHON" server.py --port "$PORT" > server.log 2>&1 &
echo $! > server.pid
echo "Started (pid $!). Waiting for warm-up..."
for _ in $(seq 1 60); do
    if grep -q "ready on" server.log; then echo "Ready: http://$(hostname -I | awk '{print $1}'):$PORT"; exit 0; fi
    if ! kill -0 "$(cat server.pid)" 2>/dev/null; then echo "Server exited:"; tail -20 server.log; exit 1; fi
    sleep 1
done
echo "Still warming up; check server.log"
