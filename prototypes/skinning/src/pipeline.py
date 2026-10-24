import subprocess
import sys
from pathlib import Path

# Find the root directory
ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"[RUNNING] {script_path.name}")
    print(f"{'='*50}")
    
    # Run the script using the current Python environment and forward any arguments
    cmd = [sys.executable, str(script_path)]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    result = subprocess.run(cmd)
    
    # Check if the script failed
    if result.returncode != 0:
        print(f"\n[ERROR] Error encountered while running {script_path.name}. Halting pipeline.")
        sys.exit(1)

def main():
    print("Initializing Data Pipeline...")
    
    # Define the strict execution order
    pipeline_steps = [
        SCRIPTS_DIR / "extract.py",
        SCRIPTS_DIR / "smoother.py",
        SCRIPTS_DIR / "converter.py" # (This is the npz_to_json script you renamed)
    ]

    # Verify all scripts exist before starting
    for script in pipeline_steps:
        if not script.exists():
            print(f"[ERROR] Cannot find {script}. Check your file paths.")
            sys.exit(1)

    # Execute them one by one
    for script in pipeline_steps:
        run_script(script)

    # Copy manual correction file from keyframes_corrected to output folder if video name is provided
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
        src_dir = ROOT_DIR / "src"
        output_dir = src_dir / "output"
        corrected_dir = src_dir / "intepolation_code" / "keyframes_corrected"
        
        manual_filename = f"{video_name}_manual.json"
        corrected_path = corrected_dir / manual_filename
        output_path = output_dir / manual_filename
        
        if corrected_path.exists():
            import shutil
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(corrected_path, output_path)
            print(f"[PIPELINE] Copied manual file from {corrected_path.relative_to(ROOT_DIR)} to {output_path.relative_to(ROOT_DIR)}")

    print("\n" + "="*50)
    print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
    print("You can now run: python prototype/visualize.py")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()