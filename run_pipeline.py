import subprocess
import sys
import os

def run_script(command_args):
    script_name = command_args[0]
    print(f"\n{'='*50}\nRunning {' '.join(command_args)}\n{'='*50}\n")
    if not os.path.exists(script_name):
        print(f"Error: {script_name} not found.")
        sys.exit(1)
        
    result = subprocess.run([sys.executable] + command_args)
    if result.returncode != 0:
        print(f"\nPipeline failed at {script_name} with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("Starting End-to-End CTR-GCN Pipeline...")
    # Pass --target 50 to the collection phase
    scripts = [
        ["phase_1_2_collect.py", "--target", "50"],
        ["phase_3_4_extract.py"],
        ["phase_5_split.py"],
        ["phase_6_7_train.py"],
        ["phase_8_9_export.py"]
    ]
    
    for script_args in scripts:
        run_script(script_args)
        
    print("\nPipeline completed successfully!")
