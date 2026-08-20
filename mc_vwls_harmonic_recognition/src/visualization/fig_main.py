import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def plot_state_space():

    data_path = (
        Path(__file__).parents[2]
        /
        "data"
        /
        "generated"
        /
        "ideal"
    )


    save_path = (
        Path(__file__).parents[2]
        /
        "results"
        /
        "figures"
        /
        "Fig1_state_space.png"
    )


    meta = pd.read_csv(
        data_path / "metadata.csv"
    )


    fig, axes = plt.subplots(
        4,
        2,
        figsize=(8,12)
    )


    for i, ell in enumerate([1,2,3,4]):


        row = meta[
            (meta["ell"]==ell)
            &
            (meta["phase_bin"]==0)
        ].iloc[0]


        state_id = int(row["state"])


        field = np.load(
            data_path /
            f"state_{state_id:03d}.npy"
        )


        intensity = np.abs(field)**2

        phase = np.angle(field)



        axes[i,0].imshow(
            intensity,
            cmap="hot"
        )

        axes[i,0].set_title(
            f"$l={ell}$ Intensity"
        )


        axes[i,1].imshow(
            phase,
            cmap="twilight"
        )

        axes[i,1].set_title(
            f"$l={ell}$ Phase"
        )


        axes[i,0].axis("off")
        axes[i,1].axis("off")


    plt.tight_layout()


    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        save_path
    )



if __name__=="__main__":

    plot_state_space()