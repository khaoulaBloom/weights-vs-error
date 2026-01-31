# Weights vs Error — Linear Regression Loss Intuition

This mini-project builds intuition for linear regression by showing how changing the parameters w (weight/slope) and b (bias/intercept) affects prediction error using Mean Squared Error (MSE). You will visualize prediction lines for different values of w, plot MSE vs w (loss curve), visualize the loss surface over (w, b), and see how gradient descent moves (w, b) to reduce MSE.

Project structure:

---

## Project structure
- `src/` : reusable code (data, model, metrics, plots, training)
- `notebooks/` : notebooks to run the experiments
- `outputs/figures/` : saved figures (optional)
- `data/` : placeholder for datasets (optional)


--- 


Setup (create a virtual environment):
Mac/Linux:
python -m venv .venv
source .venv/bin/activate

Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:
pip install -r requirements.txt

Run:
Start Jupyter:
jupyter notebook
(or: jupyter lab)

Open and run notebooks in order:
notebooks/01_weights_affect_error.ipynb
notebooks/02_gradient_descent_path.ipynb

Notebook 01_weights_affect_error:
- Generates synthetic data
- Plots prediction lines for multiple w values (with fixed b)
- Plots MSE vs w and prints the best w from the sweep

Notebook 02_gradient_descent_path:
- Computes MSE surface over (w, b)
- Runs gradient descent and shows MSE decreasing over steps and the optimization path on the MSE contour map

Saving figures:
Create the folder once:
Mac/Linux: mkdir -p outputs/figures
Windows (PowerShell): mkdir outputs\figures

In the notebook, create fig_dir (add this cell near the top, after the project_root cell):
from pathlib import Path
fig_dir = Path(project_root) / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
fig_dir

Recommended saving method (use save_path inside the plotting functions in src/visualize.py):
plot_mse_vs_w(w_values, mse_values, save_path=fig_dir / "01_mse_vs_w.png")
plot_mse_surface_contour(W, B, Z, save_path=fig_dir / "02_mse_surface.png")
plot_gd_path_on_surface(W, B, Z, result["history_w"], result["history_b"], save_path=fig_dir / "02_mse_surface_and_gd_path.png")

Troubleshooting:
If you get "ModuleNotFoundError: No module named 'src'", make sure your notebook includes the path setup cell that adds the project root to sys.path (the first cell in each notebook). After editing src/visualize.py, restart the notebook kernel (VS Code: Restart Kernel, Jupyter: Kernel → Restart).

If saved images are blank, it usually happens if plt.show() runs before saving. Use plotting functions that save via save_path (recommended), or save before plt.show().

Requirements:

Python 3.9+ recommended

Packages: numpy, matplotlib, jupyter
