import os 
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

ACTUAL_RUN = "run_gpu_20260924_153226"

BASE_DIR = Path(__file__).resolve().parent.parent

def plt_generator(df, dir_path, filename, loss_column, label):
    """Generate and save a plot from the DataFrame."""
    plt.figure()
    plt.plot(df["epoch"], df[loss_column], label=label)
    plt.title(label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Save the plot to the specified directory
    plot_path = os.path.join(dir_path, filename)
    plt.savefig(plot_path)
    plt.close()

def generate_artifacts(run=None):
    """Make artifacts of the training process, such as plots and logs."""
    # Create a directory for artifacts if it doesn't exist
    if run is None:
        run = ACTUAL_RUN
    artifacts_dir = BASE_DIR / "artifacts" / run / "dev" / "plots"
    os.makedirs(artifacts_dir, exist_ok=True)

    string = None

    csv_path =BASE_DIR / "artifacts" / run / "dev" / "logs" / "history.csv"

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
        
    plt_generator(df, artifacts_dir, "training_loss.png", "loss", "Training Loss")
    plt_generator(df, artifacts_dir, "validation_loss.png", "val_loss", "Validation Loss")

    print(f"Artifacts generated in '{artifacts_dir}' directory.")

if __name__ == "__main__":
    generate_artifacts()