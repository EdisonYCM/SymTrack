import sys
import argparse
import subprocess
import random

def main():
    launcher_parser = argparse.ArgumentParser(description='Launcher for training')
    launcher_parser.add_argument('--mode', type=str, default='multiple', help='Execution mode (ignored by torchrun but used for logic).')
    launcher_parser.add_argument('--nproc_per_node', type=int, default=1, help='Number of GPUs to use.')

    launcher_args, train_args = launcher_parser.parse_known_args()

    # Build the final command using torchrun
    command = [
        'torchrun',
        f'--nproc_per_node={launcher_args.nproc_per_node}',
        f'--master_port={random.randint(10000, 60000)}',
        'lib/train/run_training.py'
    ] + train_args # Pass ONLY the training-specific arguments

    print("Executing Final Command: \n" + " ".join(command))

    try:
        # Execute the command
        subprocess.run(command, check=True)
        print("\nTraining script finished successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining script failed with exit code {e.returncode}")
        exit(e.returncode)

if __name__ == "__main__":
    main()