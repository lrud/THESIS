#!/usr/bin/env python3
"""
Enhanced Training with Real-Time Monitoring
==========================================

Adds real-time GPU monitoring and detailed logging to training.

Usage:
    python cli/bin/train_with_monitoring.py jump_aware --use-multi-gpu --hidden-size 256
"""

import subprocess
import time
import threading
import json
from pathlib import Path

def monitor_gpu_training():
    """Monitor GPU usage during training."""
    print("\n" + "="*60)
    print("🔍 REAL-TIME TRAINING MONITOR")
    print("="*60)

    while True:
        try:
            # Get GPU info
            result = subprocess.run(['amd-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                gpu_lines = [line for line in lines if 'Radeon RX 7900 XT' in line]

                if gpu_lines:
                    print(f"\n⏰ {time.strftime('%H:%M:%S')} - GPU STATUS:")
                    for line in gpu_lines:
                        print(f"  {line.strip()}")

                # Check memory usage more specifically
                mem_lines = [line for line in lines if 'Mem-Uti' in line or 'VRAM_MEM' in line]
                if mem_lines:
                    print(f"  Memory: {mem_lines[0].strip()}")

            time.sleep(10)  # Update every 10 seconds

        except KeyboardInterrupt:
            print("\n\n⏹️ Monitoring stopped")
            break
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(5)

def main():
    """Main training with monitoring."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Training with real-time monitoring')
    parser.add_argument('model_type', choices=['jump_aware', 'rolling', 'differenced'])
    parser.add_argument('--monitor', action='store_true', help='Enable real-time monitoring')

    args, remaining = parser.parse_known_args()

    print("🚀 STARTING ENHANCED TRAINING WITH MONITORING")

    if args.monitor:
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_gpu_training, daemon=True)
        monitor_thread.start()

    # Build training command
    cmd = [
        '.venv/bin/python', 'cli/bin/train.py',
        args.model_type
    ] + remaining

    print(f"📋 Command: {' '.join(cmd)}")
    print("="*60)

    # Run training
    process = subprocess.Popen(cmd)

    try:
        process.wait()
        print("\n✅ Training completed!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")
        process.terminate()
        process.wait()

if __name__ == '__main__':
    main()