

import json
import csv
import sys

from beams import (
    ss_center_point,
    ss_off_center_point,
    ss_udl,
    cantilever_point,
    cantilever_udl
)


def main():
    try:


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
        
        if not spans:
            raise ValueError("Spans cannot be empty")

        if not point_loads:
            raise ValueError("Point loads cannot be empty")

        if not udl_loads:
            raise ValueError("UDL loads cannot be empty")

        if not beam_cases:
            raise ValueError("Beam cases cannot be empty")
        

        # 2. Check input values
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

        results = []
        checks = []

        
        for name in beam_cases:
            for L in spans:

                
                if name in [
                    "ss_center_point",
                    "ss_off_center_point",
                    "cantilever_point"
                ]:
                    loads = point_loads
                    unit = "N"
                else:
                    loads = udl_loads
                    unit = "N/m"

                for load in loads:
                    deflections = []

                    
                    for i in range(101):
                        x = i * L / 100

                        if name == "ss_off_center_point":
                            y = ss_off_center_point(
                                x, L, load, E, I, fraction * L
                            )
                        else:
                            y = beam_functions[name](x, L, load, E, I)

                        deflections.append(y)

                    max_y = max(deflections)

                    
                    results.append({
                        "beam_type": name,
                        "span_m": L,
                        "load_value": load,
                        "load_unit": unit,
                        "max_deflection_m": max_y
                    })

                    
                    expected = None

                    if name == "ss_center_point":
                        expected = load * L**3 / (48 * E * I)

                    elif name == "ss_udl":
                        expected = 5 * load * L**4 / (384 * E * I)

                    elif name == "cantilever_point":
                        expected = load * L**3 / (3 * E * I)

                    elif name == "cantilever_udl":
                        expected = load * L**4 / (8 * E * I)

                    if expected is not None:
                        passed = abs(max_y - expected) <= 1e-9 * abs(expected)
                        checks.append((name, passed))

                    
                    if name == "ss_center_point":
                        left = ss_center_point(0, L, load, E, I)
                        right = ss_center_point(L, L, load, E, I)

                        checks.append((
                            name + " boundaries",
                            abs(left) < 1e-12 and abs(right) < 1e-12
                        ))

                    elif name == "ss_off_center_point":
                        a = fraction * L

                        left = ss_off_center_point(
                            0, L, load, E, I, a
                        )
                        right = ss_off_center_point(
                            L, L, load, E, I, a
                        )

                        checks.append((
                            name + " boundaries",
                            abs(left) < 1e-12 and abs(right) < 1e-12
                        ))

                    
                    if name == "ss_center_point":
                        a = L / 2

                        left = load * a * (3 * L**2 - 4 * a**2) / (48 * E * I)
                        z = L - a
                        right = load * z * (3 * L**2 - 4 * z**2) / (48 * E * I)

                        checks.append((
                            name + " continuity",
                            abs(left - right) < 1e-12
                        ))

                    elif name == "ss_off_center_point":
                        a = fraction * L
                        b = L - a

                        left = load * b * a * (L**2 - b**2 - a**2) / (6 * L * E * I)
                        right = load * a * b * (L**2 - a**2 - b**2) / (6 * L * E * I)

                        checks.append((
                            name + " continuity",
                            abs(left - right) < 1e-12
                        ))

        
        with open("results.csv", "w", newline="") as file:
            columns = [
                "beam_type",
                "span_m",
                "load_value",
                "load_unit",
                "max_deflection_m"
            ]

            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        print("Results saved to results.csv")
        print("Total cases:", len(results))

        
        with open("results.csv", "r", newline="") as file:
            reader = csv.DictReader(file)
            saved_results = list(reader)

        
        print("\nLargest deflection for each beam type:")

        for name in beam_cases:
            largest = None

            for row in saved_results:
                if row["beam_type"] == name:
                    if largest is None:
                        largest = row
                    elif float(row["max_deflection_m"]) > float(
                        largest["max_deflection_m"]
                    ):
                        largest = row

            if largest is not None:
                print(
                    name,
                    "| span:", largest["span_m"], "m",
                    "| load:", largest["load_value"],
                    largest["load_unit"],
                    "| deflection:", largest["max_deflection_m"], "m"
                )

        
        print("\nSelf-checks:")

        failed = False

        for name, passed in checks:
            if passed:
                print(name, ": PASS")
            else:
                print(name, ": FAIL")
                failed = True

        if failed:
            print("Some checks failed.")
        else:
            print("All checks passed.")

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
