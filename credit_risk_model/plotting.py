import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

def plot_threshold_scoring(model,scoring:str='')->plt.figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        model.cv_results_["thresholds"],
        model.cv_results_["scores"],
        marker="o",
        linewidth=1e-3,
        markersize=4,
        color="#c0c0c0",
    )
    ax.plot(
        model.best_threshold_,
        model.best_score_,
        "^",
        markersize=10,
        color="#ff6700",
        label=f"Optimal cut-off point = {model.best_threshold_:.2f}",
    )
    ax.legend(fontsize=8, loc="lower center")
    ax.set_xlabel("Decision threshold", fontsize=10)
    ax.set_ylabel(f"{scoring} score", fontsize=10)
    ax.set_title(f"{scoring} score vs. Decision threshold -- Cross-validation", fontsize=12)
    return fig