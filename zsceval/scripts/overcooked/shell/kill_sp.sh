#!/bin/bash
# Kill all train_sp.py related processes

LAYOUT=$1

if [ -z "$LAYOUT" ]; then
    echo "Killing all train_sp processes..." >&2
    # 1. Kill parent bash scripts first to prevent spawning new processes
    pkill -9 -f "train_sp.sh" 2>/dev/null
    # 2. Kill Python train processes
    pkill -9 -f "train/train_sp.py" 2>/dev/null
    # 3. Kill worker processes
    pkill -9 -f "mappo-Overcooked.*-sp" 2>/dev/null
else
    echo "Killing train_sp processes for layout: ${LAYOUT}..." >&2
    # 1. Kill parent bash scripts first to prevent spawning new processes
    pkill -9 -f "train_sp.sh.*${LAYOUT}" 2>/dev/null
    # 2. Kill Python train processes
    pkill -9 -f "train/train_sp.py.*${LAYOUT}" 2>/dev/null
    pkill -9 -f "layout_name ${LAYOUT}" 2>/dev/null
    # 3. Kill worker processes
    pkill -9 -f "mappo-Overcooked.*${LAYOUT}.*-sp" 2>/dev/null
fi

sleep 1

# Check remaining
REMAINING_PARENT=$(ps aux | grep "train_sp.sh" | grep -v grep 2>/dev/null)
REMAINING_TRAIN=$(ps aux | grep "train/train_sp.py" | grep -v grep 2>/dev/null)
REMAINING_WORKERS=$(ps aux | grep "mappo-Overcooked.*-sp" | grep -v grep 2>/dev/null)

echo "" >&2
if [ -z "$REMAINING_PARENT" ] && [ -z "$REMAINING_TRAIN" ] && [ -z "$REMAINING_WORKERS" ]; then
    echo "✓ All processes killed" >&2
else
    if [ ! -z "$REMAINING_PARENT" ]; then
        echo "⚠ Remaining parent scripts:" >&2
        echo "$REMAINING_PARENT" >&2
    fi
    if [ ! -z "$REMAINING_TRAIN" ]; then
        echo "⚠ Remaining train_sp:" >&2
        echo "$REMAINING_TRAIN" >&2
    fi
    if [ ! -z "$REMAINING_WORKERS" ]; then
        echo "⚠ Remaining workers:" >&2
        echo "$REMAINING_WORKERS" >&2
    fi
fi

