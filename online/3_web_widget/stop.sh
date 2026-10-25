#!/bin/bash
# Stop the ISL demo server started by start.sh.
cd "$(dirname "$0")"
if [ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null; then
    echo "Stopped (pid $(cat server.pid))."
else
    echo "Not running."
fi
rm -f server.pid
