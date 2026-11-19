
import numpy as np
import pandas as pd
from beams import ss_center_point


np.random.seed(42)

data = []

for beam_id in range(500):

    L = np.random.uniform(2, 6)           
    E = np.random.uniform(70e9, 210e9)    
    I = np.random.uniform(1e-6, 10e-6)
    P = np.random.uniform(1000, 10000)   
    # Generate 21 positions along the beam
    x_values = np.linspace(0, L, 21)

    # Calculate deflection using beams.py
    y_values = ss_center_point(x_values, L, P, E, I)

    # Store each position and its deflection
    for x, y in zip(x_values, y_values):

        data.append([beam_id, x, L, E, I, P, y])



df = pd.DataFrame(data, columns=[
    "beam_id", "x", "L", "E", "I", "P", "y"
])


df.to_csv("beam_data.csv", index=False)

print("Dataset generated successfully!")
print("Dataset shape:", df.shape)
print(df.head())



