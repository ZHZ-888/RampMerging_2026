#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_following_state_cm(cm, accuracy, free_recall, macro_f1,
                            save_path=None, dpi=600, figsize=(5, 4)):
    cm = np.asarray(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    custom_blues = LinearSegmentedColormap.from_list("custom_blues", ["white", "#0000FF"])
    class_names = ["0", "1"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=custom_blues, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted state", fontsize=11)
    ax.set_ylabel("True state", fontsize=11)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]})"
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=11, color=text_color)

    metric_text = (
        f"Accuracy    : {accuracy:.3f}\n"
        f"Free recall : {free_recall:.3f}\n"
        f"Macro F1    : {macro_f1:.3f}"
    )

    ax.text(
        0.98, 0.02, metric_text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(facecolor="white", edgecolor="black",
                  boxstyle="square,pad=0.25", linewidth=0.8)
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path).with_suffix(".tiff")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path, format="tiff", dpi=dpi, bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"},
        )

    plt.show()

#%%
cm = [[1159, 49],
      [94, 3776]]

accuracy = 0.9718393068137062
free_recall = 0.959
macro_f1 = 0.962   # 换成你实际计算出的完整值

if __name__ == "__main__":
    if "__file__" in globals():
        root = Path(__file__).resolve().parents[2]
    else:
        cwd = Path.cwd()
        root = next(
            (folder for folder in (cwd, *cwd.parents)
             if (folder / "run.py").is_file() and (folder / "plot").is_dir()),
            None,
        )
        if root is None:
            raise FileNotFoundError("Open the IPython console in the project folder or one of its subfolders.")
    save_path = root / "plot" / "plot_figures" / "following_state_rf.tiff"
    plot_following_state_cm(cm, accuracy, free_recall, macro_f1, save_path=save_path)
