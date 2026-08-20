"""
Von Karman turbulence phase screen generator with cache
OAM_MC_VWLS_Project
"""

import numpy as np
from pathlib import Path

from src.physics.lg_mode import load_config



# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

CACHE_DIR = (
    ROOT
    /
    "data"
    /
    "cache"
    /
    "turbulence"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ==========================================================
# Frequency grid
# ==========================================================

def frequency_grid(size, delta):

    fx = np.fft.fftfreq(
        size,
        d=delta
    )

    fy = np.fft.fftfreq(
        size,
        d=delta
    )


    FX, FY = np.meshgrid(
        fx,
        fy
    )


    FR = np.sqrt(
        FX**2 + FY**2
    )


    return FX, FY, FR



# ==========================================================
# Fried parameter
# ==========================================================

def fried_parameter(
        Cn2,
        k,
        dz):

    return (
        0.423
        *
        k**2
        *
        Cn2
        *
        dz
    ) ** (-3/5)



# ==========================================================
# Generate phase screen
# ==========================================================

def generate_phase_screen(
        Cn2,
        dz,
        seed=0):


    cfg = load_config()


    N = cfg["optics"]["grid_size"]

    window = cfg["optics"]["window_size"]

    wavelength = float(
        cfg["optics"]["wavelength"]
    )


    L0 = cfg["turbulence"]["outer_scale"]

    l0 = cfg["turbulence"]["inner_scale"]


    dx = window / N

    delta_f = 1/window


    k = 2*np.pi/wavelength


    r0 = fried_parameter(
        Cn2,
        k,
        dz
    )


    _,_,FR = frequency_grid(
        N,
        dx
    )


    f0 = 1/L0

    fm = 5.92/l0/(2*np.pi)


    PSD = (
        0.023
        *
        r0**(-5/3)
        *
        np.exp(
            -(FR/fm)**2
        )
        /
        (
            (FR**2+f0**2)
            **
            (11/6)
        )
    )


    PSD[0,0]=0


    rng = np.random.default_rng(
        seed
    )


    noise = (
        rng.standard_normal(
            (N,N)
        )
        +
        1j*
        rng.standard_normal(
            (N,N)
        )
    )


    spectrum = (
        noise
        *
        np.sqrt(PSD)
        *
        delta_f
    )


    phase = np.fft.ifft2(
        spectrum
    )


    phase = np.real(
        phase
    )


    phase *= N**2


    phase -= np.mean(
        phase
    )


    return phase.astype(
        np.float32
    )



# ==========================================================
# Cache interface
# ==========================================================

def get_phase_screen(
        Cn2,
        dz,
        seed=0):


    name = (
        f"Cn2_{Cn2:.1e}"
        f"_dz_{dz:.1f}"
        f"_{seed:04d}.npy"
    )


    path = CACHE_DIR / name



    if path.exists():

        phase = np.load(
            path
        )


    else:

        phase = generate_phase_screen(
            Cn2,
            dz,
            seed
        )


        np.save(
            path,
            phase
        )


    return phase



# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":


    cfg = load_config()


    Cn2 = cfg["turbulence"]["Cn2"][0]


    distance = (
        cfg["turbulence"]
        ["propagation_distance"][0]
    )


    screen_number = (
        cfg["turbulence"]
        ["phase_screen_number"]
    )


    dz = distance/screen_number



    print("="*50)

    print(
        "Generating turbulence cache"
    )

    print("="*50)



    for i in range(5):

        phase = get_phase_screen(
            Cn2,
            dz,
            seed=i
        )


        print(
            i,
            phase.shape,
            np.std(phase)
        )



    print()

    print(
        "Cache path:"
    )

    print(
        CACHE_DIR
    )