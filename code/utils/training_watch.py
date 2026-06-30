"""Background training monitor — runs as a self-loop, logs periodic status.

Usage:
    # Step 1: Start training (in one terminal)
    python code/my_ai/es_train.py --generations 200 --workers 8 --out-dir training_history_test

    # Step 2: Start monitoring (in another terminal)
    python code/utils/training_watch.py --dir training_history_test --interval 1800

    # Step 3: Go to sleep. Come back and ask the AI "训练怎样了"
    #         The AI reads the monitor log and summarizes.

The monitor loops every N seconds, reads the latest training log