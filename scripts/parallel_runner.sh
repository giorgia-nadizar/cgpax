#!/usr/bin/env bash

SESSION="gpax_runs"

# Start the session in detached mode with a base window (optional)
byobu new-session -d -s $SESSION -n base
byobu send-keys -t $SESSION:base "echo 'GPAX run session started'" C-m

FOSS=("U" "RT" "LT" "LT1")

# Loop over seeds
for SEED in {0..9}; do
  # Loop over FOS modes
  for FOS in "${FOSS[@]}"; do
    WIN_NAME="seed_${SEED}_${FOS}"

    # Create a new window
    byobu new-window -t $SESSION -n "$WIN_NAME"

    # Send commands to the window
    byobu send-keys -t $SESSION:"$WIN_NAME" "conda activate gpax" C-m
    byobu send-keys -t $SESSION:"$WIN_NAME" "export PYTHONPATH=\$PWD" C-m
    byobu send-keys -t $SESSION:"$WIN_NAME" "cd experiments" C-m
    byobu send-keys -t $SESSION:"$WIN_NAME" "python run_gomea.py --seed $SEED --fos_mode $FOS --n_individuals 100" C-m
    byobu send-keys -t $SESSION:"$WIN_NAME" "python run_gomea.py --seed $SEED --fos_mode $FOS --n_individuals 1000" C-m
  done
done

# Loop over seeds
for SEED in {0..9}; do
  WIN_NAME="seed_$SEED"

  # Create a new window
  byobu new-window -t $SESSION -n "$WIN_NAME"

  # Send the series of commands to run in the window
  byobu send-keys -t $SESSION:"$WIN_NAME" "conda activate gpax" C-m
  byobu send-keys -t $SESSION:"$WIN_NAME" "export PYTHONPATH=\$PWD" C-m
  byobu send-keys -t $SESSION:"$WIN_NAME" "cd experiments" C-m
  byobu send-keys -t $SESSION:"$WIN_NAME" "python run_ga.py --seed $SEED --n_individuals 100" C-m
  byobu send-keys -t $SESSION:"$WIN_NAME" "python run_ga.py --seed $SEED --n_individuals 1000" C-m
done

# Attach to the session
byobu attach-session -t $SESSION
