import os
import sys
import subprocess
import time

def run_app():
    print("=" * 65)
    print(" 🚀 Launching TerraScope AI Pipeline & Dashboard ")
    print(" Semantic Satellite Search and Environmental Change Intelligence ")
    print("=" * 65)

    # 1. Initialize dataset if not present
    from terrascope.data_generator import create_dataset_and_seed_db
    if not os.path.exists("data/bengaluru_tiles"):
        print("Generating study area Sentinel-2 dataset around Bengaluru...")
        create_dataset_and_seed_db()

    print("\nStarting Streamlit Application on http://localhost:8501 ...")
    
    # Run Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.headless=true"])

if __name__ == "__main__":
    run_app()
