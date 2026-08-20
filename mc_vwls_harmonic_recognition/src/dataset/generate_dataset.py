"""
Batch GPU HDF5 Dataset Generator

OAM_MC_VWLS_Project
"""

import numpy as np
import h5py

from pathlib import Path
from tqdm import tqdm


from src.physics.lg_mode import (
    load_cached_lg_modes,
    load_config
)

from src.physics.turbulence import (
    propagate_through_turbulence,
    apply_occlusion
)



ROOT = Path(__file__).resolve().parents[2]


SAVE_PATH = (
    ROOT
    /
    "data"
    /
    "generated"
    /
    "turbulence_mask_v1.h5"
)


BATCH_SIZE = 64



def generate_dataset():


    cfg = load_config()


    states = load_cached_lg_modes()


    Cn2_list = cfg["turbulence"]["Cn2"]

    distance_list = cfg["turbulence"]["propagation_distance"]

    mask_list = cfg["mask"]["energy_ratio"]


    repeat = cfg["dataset"]["realizations_per_condition"]



    total = (

        len(states)
        *
        len(Cn2_list)
        *
        len(distance_list)
        *
        len(mask_list)
        *
        repeat

    )



    print(
        "Total samples:",
        total
    )



    SAVE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )



    h5 = h5py.File(
        SAVE_PATH,
        "w"
    )



    fields = h5.create_dataset(

        "fields",

        shape=(
            total,
            256,
            256
        ),

        dtype=np.complex64,

        chunks=(
            BATCH_SIZE,
            256,
            256
        ),

        compression="gzip",

        compression_opts=4

    )



    labels = h5.create_dataset(

        "labels",

        shape=(total,),

        dtype=np.int32

    )



    conditions = h5.create_dataset(

        "conditions",

        shape=(total,4),

        dtype=np.float32

    )



    buffer_field = []

    buffer_label = []

    buffer_cond = []



    index = 0



    def flush():

        nonlocal index


        if len(buffer_field)==0:

            return



        n = len(buffer_field)



        fields[
            index:index+n
        ] = np.asarray(
            buffer_field
        )



        labels[
            index:index+n
        ] = np.asarray(
            buffer_label
        )



        conditions[
            index:index+n
        ] = np.asarray(
            buffer_cond
        )



        index += n


        buffer_field.clear()

        buffer_label.clear()

        buffer_cond.clear()




    with tqdm(total=total) as bar:


        for state_id, field in enumerate(states):


            for Cn2 in Cn2_list:


                for distance in distance_list:


                    for mask_ratio in mask_list:


                        for seed in range(repeat):


                            out = propagate_through_turbulence(

                                field,

                                Cn2=Cn2,

                                distance=distance,

                                screen_number=
                                cfg["turbulence"]["phase_screen_number"],

                                seed=seed

                            )



                            out = apply_occlusion(

                                out,

                                mask_ratio

                            )



                            buffer_field.append(

                                out.astype(
                                    np.complex64
                                )

                            )


                            buffer_label.append(

                                state_id

                            )


                            buffer_cond.append(

                                [

                                    Cn2,

                                    distance,

                                    mask_ratio,

                                    seed

                                ]

                            )



                            if len(buffer_field)>=BATCH_SIZE:

                                flush()



                            bar.update(1)



    flush()


    h5.close()



    print(
        "Saved:",
        SAVE_PATH
    )


    print(
        "Samples:",
        index
    )



if __name__=="__main__":

    generate_dataset()