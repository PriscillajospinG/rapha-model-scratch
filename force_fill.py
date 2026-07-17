import os
import subprocess
import time

CLASSES = ["ankle", "calf", "hamstring", "heel_slide", "hip", "knee", "leg_raise", "quadriceps", "toes"]
TARGET = 50

def check_counts():
    counts = {}
    for c in CLASSES:
        d = os.path.join("datasets/lower_limb/raw", c)
        if os.path.exists(d):
            counts[c] = len([f for f in os.listdir(d) if f.endswith('.mp4')])
        else:
            counts[c] = 0
    return counts

iteration = 1
while True:
    counts = check_counts()
    print(f"--- Iteration {iteration} Counts ---")
    all_done = True
    for c, count in counts.items():
        print(f"{c}: {count}/{TARGET}")
        if count < TARGET:
            all_done = False
            
    if all_done:
        print("All classes have reached 50 videos!")
        break
        
    print("\nRunning collection phase...")
    subprocess.run(["python", "phase_1_2_collect.py", "--target", "50"])
    
    print("\nRunning deduplication phase...")
    subprocess.run(["python", "datasets/lower_limb/clean_dataset.py"])
    
    print("\nRebuilding metadata...")
    subprocess.run(["python", "fix_metadata.py"])
    
    iteration += 1
