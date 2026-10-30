import numpy as np
import matplotlib.pyplot as plt

from beams import (
    ss_center_point,
    ss_off_center_point,
    ss_udl,
    cantilever_point,
    cantilever_udl
)


def create_plots(config):

    E = config["E"]
    I = config["I"]
    spans = config["spans"]

    beam_functions = {
        "ss_center_point": ss_center_point,
        "ss_off_center_point": ss_off_center_point,
        "ss_udl": ss_udl,
        "cantilever_point": cantilever_point,
        "cantilever_udl": cantilever_udl
    }

    for name in config["beam_cases"]:

        
        if name in ["ss_center_point", "ss_off_center_point",
                    "cantilever_point"]:
            loads = config["point_loads"]
        else:
            loads = config["udl_loads"]

        
        load = loads[len(loads) // 2]

        
        plt.figure()

        for L in spans:

            
            x = np.linspace(0, L, 101)

            
            if name == "ss_off_center_point":
                a = config["off_center_fraction"] * L
                y = beam_functions[name](x, L, load, E, I, a)
            else:
                y = beam_functions[name](x, L, load, E, I)

            
            plt.plot(x, y, label=f"Span = {L} m")

            
            max_index = np.argmax(np.abs(y))
            plt.plot(x[max_index], y[max_index], "ro")


        plt.title(name.replace("_", " ").title())
        plt.xlabel("Position (m)")
        plt.ylabel("Deflection (m)")
        plt.legend()
        plt.grid()

        
        plt.gca().invert_yaxis()

        
        plt.savefig(f"outputs/{name}.png")
        plt.close()

        print(f"Plot: outputs/{name}.png")