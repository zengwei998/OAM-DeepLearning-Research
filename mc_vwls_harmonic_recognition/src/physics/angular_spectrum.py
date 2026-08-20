"""
GPU Accelerated Angular Spectrum Propagation

OAM_MC_VWLS_Project
"""

import numpy as np
import torch

from src.physics.lg_mode import load_config



# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)



print(
    "Angular Spectrum Device:",
    DEVICE
)



# ==========================================================
# Transfer function cache
# ==========================================================

_H_CACHE = {}



def get_transfer_function(
        N,
        wavelength,
        window,
        distance):


    key = (
        N,
        wavelength,
        window,
        distance,
        str(DEVICE)
    )


    if key in _H_CACHE:

        return _H_CACHE[key]



    dx = window / N


    fx = np.fft.fftfreq(
        N,
        d=dx
    )

    fy = np.fft.fftfreq(
        N,
        d=dx
    )


    FX, FY = np.meshgrid(
        fx,
        fy
    )


    k = (
        2
        *
        np.pi
        /
        wavelength
    )


    argument = (

        1

        -

        (wavelength*FX)**2

        -

        (wavelength*FY)**2

    )


    H = np.exp(

        1j
        *
        k
        *
        distance
        *
        np.sqrt(
            np.maximum(argument,0)
        )

    )


    H = torch.tensor(
        H,
        dtype=torch.complex64,
        device=DEVICE
    )



    _H_CACHE[key] = H


    return H



# ==========================================================
# Angular spectrum propagation
# ==========================================================

def angular_spectrum_propagation(
        field,
        distance):


    cfg = load_config()


    wavelength = float(
        cfg["optics"]["wavelength"]
    )

    N = cfg["optics"]["grid_size"]

    window = cfg["optics"]["window_size"]



    H = get_transfer_function(

        N,

        wavelength,

        window,

        distance

    )



    # numpy -> torch

    field_t = torch.tensor(

        field,

        dtype=torch.complex64,

        device=DEVICE

    )



    spectrum = torch.fft.fft2(
        field_t
    )


    propagated = torch.fft.ifft2(

        spectrum
        *
        H

    )



    # cuda -> numpy

    result = (

        propagated

        .detach()

        .cpu()

        .numpy()

    )


    return result



# ==========================================================
# Apply phase screen
# ==========================================================

def apply_phase_screen(
        field,
        phase):


    return (

        field

        *

        np.exp(
            1j*phase
        )

    )



# ==========================================================
# Test
# ==========================================================

if __name__=="__main__":


    from src.physics.lg_mode import load_cached_lg_modes


    print("="*50)

    print(
        "Angular spectrum test"
    )

    print("="*50)



    states = load_cached_lg_modes()


    field = states[1]


    print(
        "Input power:",
        np.sum(
            np.abs(field)**2
        )
    )



    output = angular_spectrum_propagation(

        field,

        500

    )


    print(

        "Output shape:",

        output.shape

    )



    print(

        "Output power:",

        np.sum(
            np.abs(output)**2
        )

    )