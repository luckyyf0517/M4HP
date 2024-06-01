#!/bin/bash

LOG_FILE="output.log"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 command [arguments...]"
    exit 1
fi

if [ -f "$LOG_FILE" ]; then
    echo "Removing existing log file: $LOG_FILE"
    rm -f "$LOG_FILE"
fi

COMMAND="$@"

echo "Running command: $COMMAND"
nohup $COMMAND > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"

./view.sh