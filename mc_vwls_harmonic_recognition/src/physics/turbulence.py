"""
Turbulence channel model
with cached Von Karman phase screens

OAM_MC_VWLS_Project
"""

import numpy as np

from src.physics.phase_screen import get_phase_screen

from src.physics.angular_spectrum import (
    angular_spectrum_propagation,
    apply_phase_screen
)



# ==========================================================
# Turbulence propagation
# ==========================================================

def propagate_through_turbulence(
        field,
        Cn2,
        distance,
        screen_number=8,
        seed=0):


    dz = distance / screen_number


    output = field.copy()



    for i in range(screen_number):


        phase = get_phase_screen(
            Cn2=Cn2,
            dz=dz,
            seed=seed+i
        )


        output = apply_phase_screen(
            output,
            phase
        )


        output = angular_spectrum_propagation(
            output,
            dz
        )


    return output



# ==========================================================
# Circular occlusion mask
# ==========================================================

def circular_mask(
        size,
        ratio,
        position=None):


    yy, xx = np.indices(
        (size, size)
    )


    if position is None:

        cx = size // 2
        cy = size // 2

    else:

        cx, cy = position



    radius = (
        np.sqrt(ratio)
        *
        size
        /
        2
    )



    mask = (
        (xx-cx)**2
        +
        (yy-cy)**2
        <=
        radius**2
    )


    return mask.astype(
        np.float32
    )



# ==========================================================
# Apply occlusion
# ==========================================================

def apply_occlusion(
        field,
        energy_ratio,
        position=None):


    if energy_ratio >= 1.0:

        return field



    N = field.shape[0]


    mask = circular_mask(
        N,
        energy_ratio,
        position
    )


    return field * mask



# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":


    from src.physics.lg_mode import load_cached_lg_modes



    print("="*50)

    print(
        "Turbulence channel test"
    )

    print("="*50)



    # load cached LG states

    states = load_cached_lg_modes()


    field = states[2]



    print(
        "Input power:",
        np.sum(
            np.abs(field)**2
        )
    )



    output = propagate_through_turbulence(

        field,

        Cn2=1e-14,

        distance=500,

        screen_number=8,

        seed=0

    )



    print(
        "After turbulence:",
        np.sum(
            np.abs(output)**2
        )
    )



    masked = apply_occlusion(

        output,

        energy_ratio=0.4

    )



    print(
        "After mask:",
        np.sum(
            np.abs(masked)**2
        )
    )


    print(
        "Output shape:",
        masked.shape
    )