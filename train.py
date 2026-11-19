import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv("beam_data.csv")

print(df.head())
print("Dataset shape:", df.shape)

beam_ids = df["beam_id"].unique()


np.random.seed(42)
np.random.shuffle(beam_ids)

train_ids = beam_ids[:350]
val_ids = beam_ids[350:425]
test_ids = beam_ids[425:]


train_df = df[df["beam_id"].isin(train_ids)].copy()
val_df = df[df["beam_id"].isin(val_ids)].copy()
test_df = df[df["beam_id"].isin(test_ids)].copy()

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)


# Select input columns and target column

input_columns = ["x", "L", "E", "I", "P"]

X_train = train_df[input_columns].to_numpy()
X_val = val_df[input_columns].to_numpy()
X_test = test_df[input_columns].to_numpy()

y_train = train_df[["y"]].to_numpy()
y_val = val_df[["y"]].to_numpy()
y_test = test_df[["y"]].to_numpy()


# Calculate mean and standard deviation from training data

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)


# Normalize inputs

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


# Normalize outputs

y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std


print("Normalization completed!")
print("Training input shape:", X_train.shape)
print("Training output shape:", y_train.shape)
print("Input means:", X_mean)
print("Input standard deviations:", X_std)# Convert data into PyTorch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Data converted into tensors!")



train_data = TensorDataset(X_train, y_train)



train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

print("Number of batches:", len(train_loader))

# Build the neural network

model = nn.Sequential(
    nn.Linear(5, 64),
    nn.ReLU(),

    nn.Linear(64, 64),
    nn.ReLU(),

    nn.Linear(64, 1)
)

print(model)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

epochs = 100
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(epochs):

    model.train()
    total_train_loss = 0

    for X_batch, y_batch in train_loader:

        predictions = model(X_batch)

        loss = loss_function(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)

    # Calculate validation loss

    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_function(val_predictions, y_val)

    train_losses.append(train_loss)
    val_losses.append(val_loss.item())
        # Save the best model

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()
        best_epoch = epoch + 1

        torch.save(model.state_dict(), "best_model.pth")

    print("Epoch:", epoch + 1,
          "Training Loss:", train_loss,
          "Validation Loss:", val_loss.item())

print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)

# Load the best model

model.load_state_dict(torch.load("best_model.pth", weights_only=True))

model.eval()

print("Best model loaded successfully!")



with torch.no_grad():
    predictions = model(X_test)

print("Predictions generated!")



predictions = predictions.numpy() * y_std + y_mean
actual = y_test.numpy() * y_std + y_mean



mae = np.mean(np.abs(actual - predictions))

rmse = np.sqrt(np.mean((actual - predictions) ** 2))


relative_l2 = np.linalg.norm(actual - predictions) / np.linalg.norm(actual)

print("Test MAE:", mae * 1000, "mm")
print("Test RMSE:", rmse * 1000, "mm")
print("Relative L2 Error:", relative_l2)
import matplotlib.pyplot as plt

# Plot training and validation loss

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.grid()

plt.savefig("learning_curve.png")
plt.show()

# Plot predicted vs actual deflection

plt.figure()

plt.scatter(actual * 1000, predictions * 1000)

# Reference line
minimum = np.min(actual * 1000)
maximum = np.max(actual * 1000)

plt.plot([minimum, maximum], [minimum, maximum], "r--")

plt.xlabel("Actual Deflection (mm)")
plt.ylabel("Predicted Deflection (mm)")
plt.title("Predicted vs Actual Deflection")

plt.grid()
plt.savefig("predicted_vs_actual.png")
plt.show()



errors = (predictions - actual) * 1000


plt.figure()

plt.hist(errors, bins=30, edgecolor="black")

plt.xlabel("Prediction Error (mm)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")

plt.grid()
plt.savefig("error_histogram.png")
plt.show()


test_df["predicted_y"] = predictions.flatten()



test_beams = test_df["beam_id"].unique()[:3]



# Plot actual and predicted deflection curves

for beam_id in test_beams:

    beam_data = test_df[test_df["beam_id"] == beam_id]
    beam_data = beam_data.sort_values("x")

    plt.figure()

    plt.plot(
        beam_data["x"],
        beam_data["y"] * 1000,
        label="Actual"
    )

    plt.plot(
        beam_data["x"],
        beam_data["predicted_y"] * 1000,
        "--",
        label="Predicted"
    )

    plt.xlabel("Position along beam (m)")
    plt.ylabel("Deflection (mm)")
    plt.title("Beam " + str(beam_id) + " Deflection")

    plt.legend()
    plt.grid()

    plt.savefig("beam_" + str(beam_id) + "_comparison.png")
    plt.show()