
import json
import sys
import os
import numpy as np
import pandas as pd
from plot import create_plots

from beams import (
    ss_center_point,
    ss_off_center_point,
    ss_udl,
    cantilever_point,
    cantilever_udl
)


def main():
    try:
        # 1. Read configuration
        config_file = "config.json"

        if len(sys.argv) > 1:
            config_file = sys.argv[1]

        with open(config_file, "r") as file:
            config = json.load(file)

        E = config["E"]
        I = config["I"]
        spans = config["spans"]
        point_loads = config["point_loads"]
        udl_loads = config["udl_loads"]
        fraction = config["off_center_fraction"]
        beam_cases = config["beam_cases"]

        # 2. Validate inputs
        if not spans:
            raise ValueError("Spans cannot be empty")

        if not point_loads:
            raise ValueError("Point loads cannot be empty")

        if not udl_loads:
            raise ValueError("UDL loads cannot be empty")

        if not beam_cases:
            raise ValueError("Beam cases cannot be empty")

        if E <= 0 or I <= 0:
            raise ValueError("E and I must be positive")

        for L in spans:
            if L <= 0:
                raise ValueError("Span lengths must be positive")

        for P in point_loads:
            if P <= 0:
                raise ValueError("Point loads must be positive")

        for w in udl_loads:
            if w <= 0:
                raise ValueError("UDL loads must be positive")

        if not 0 < fraction < 1:
            raise ValueError("Off-center fraction must be between 0 and 1")

        # 3. Select beam functions
        beam_functions = {
            "ss_center_point": ss_center_point,
            "ss_off_center_point": ss_off_center_point,
            "ss_udl": ss_udl,
            "cantilever_point": cantilever_point,
            "cantilever_udl": cantilever_udl
        }

        for name in beam_cases:
            if name not in beam_functions:
                raise ValueError("Unknown beam case: " + name)

        # 4. Create output folder
        os.makedirs("outputs", exist_ok=True)

        # 5. Calculate deflections and save CSV files
        for name in beam_cases:

            if name in ["ss_center_point", "ss_off_center_point",
                        "cantilever_point"]:
                loads = point_loads
                unit = "N"
            else:
                loads = udl_loads
                unit = "N/m"

            rows = []

            for L in spans:

                # Generate 101 positions
                x = np.linspace(0, L, 101)

                for load in loads:

                    # Calculate deflection using NumPy
                    if name == "ss_off_center_point":
                        y = beam_functions[name](
                            x, L, load, E, I, fraction * L
                        )
                    else:
                        y = beam_functions[name](
                            x, L, load, E, I
                        )

                    # Store results
                    for i in range(len(x)):
                        rows.append({
                            "span_m": L,
                            "load_value": load,
                            "load_unit": unit,
                            "position_m": x[i],
                            "deflection_m": y[i]
                        })

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Save one CSV for this beam type
            filename = "outputs/" + name + ".csv"
            df.to_csv(filename, index=False)

            print("Saved:", filename)

        print("\nAll CSV files generated successfully.")
        create_plots(config)

        print("All plots generated successfully.")

        
        

    except FileNotFoundError:
        print("Error: Configuration file not found.")

    except json.JSONDecodeError:
        print("Error: Configuration file contains invalid JSON.")

    except KeyError as error:
        print("Error: Missing configuration key:", error)

    except ValueError as error:
        print("Error:", error)


if __name__ == "__main__":
    main()
