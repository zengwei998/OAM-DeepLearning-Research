"""
Discrete OAM state generator with cache

OAM_MC_VWLS_Project
"""

import numpy as np
from pathlib import Path
from scipy.special import genlaguerre
import yaml



ROOT = Path(__file__).resolve().parents[2]


CONFIG_PATH = (
    ROOT /
    "config" /
    "frozen_v1.yaml"
)


CACHE_DIR = (
    ROOT /
    "data" /
    "cache" /
    "lg_modes"
)


CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)



def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)



# ==========================================================
# LG mode
# ==========================================================

def generate_lg_mode(
        l,
        p,
        waist,
        grid_size,
        window_size):


    x = np.linspace(
        -window_size/2,
        window_size/2,
        grid_size
    )


    y = np.linspace(
        -window_size/2,
        window_size/2,
        grid_size
    )


    X,Y = np.meshgrid(
        x,
        y
    )


    R = np.sqrt(
        X**2+Y**2
    )


    theta = np.arctan2(
        Y,
        X
    )


    rho = np.sqrt(2)*R/waist


    L = genlaguerre(
        p,
        abs(l)
    )(rho**2)


    amplitude = (

        rho**abs(l)

        *

        L

        *

        np.exp(
            -rho**2/2
        )

    )


    field = (

        amplitude

        *

        np.exp(
            1j*l*theta
        )

    )


    power = np.sum(
        np.abs(field)**2
    )


    field /= np.sqrt(power)


    return field.astype(
        np.complex64
    )



# ==========================================================
# Generate 32 discrete states
# ==========================================================

def build_lg_cache():


    cfg = load_config()


    waist = cfg["optics"]["waist"]

    N = cfg["optics"]["grid_size"]

    window = cfg["optics"]["window_size"]


    modes = cfg["oam"]["modes"]

    phase_bins = cfg["oam"]["phase_bins"]


    index = 0



    for l in modes:


        base = generate_lg_mode(

            l=l,

            p=0,

            waist=waist,

            grid_size=N,

            window_size=window

        )


        for phase_id in range(
            phase_bins
        ):


            phase = (
                2*np.pi
                *
                phase_id
                /
                phase_bins
            )


            # 离散OAM叠加态

            state = (

                base

                +

                np.exp(
                    1j*phase
                )
                *
                np.conj(base)

            )


            power = np.sum(
                np.abs(state)**2
            )


            state /= np.sqrt(power)



            path = (

                CACHE_DIR

                /

                f"lg_state_{index:03d}.npy"

            )


            np.save(
                path,
                state.astype(
                    np.complex64
                )
            )


            print(
                "Saved:",
                path.name,
                "l=",
                l,
                "phase=",
                phase_id
            )


            index += 1



def load_cached_lg_modes():


    files = sorted(
        CACHE_DIR.glob(
            "lg_state_*.npy"
        )
    )


    return [

        np.load(f)

        for f in files

    ]



# ==========================================================
# Test
# ==========================================================

if __name__=="__main__":


    print(
        "Building 32 OAM states..."
    )


    build_lg_cache()


    states = load_cached_lg_modes()


    print(
        "Total states:",
        len(states)
    )


    for i in range(3):

        print(
            i,
            states[i].shape,
            np.sum(
                np.abs(states[i])**2
            )
        )