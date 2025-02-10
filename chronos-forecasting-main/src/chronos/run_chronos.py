
import pandas as pd  
import numpy
import torch
from chronos import BaseChronosPipeline, ChronosPipeline, ChronosBoltPipeline
import matplotlib.pyplot as plt 

# Load the Chronos model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # Use "amazon/chronos-bolt-small" for Bolt model
    device_map="cpu",  # Change to "cuda" if using GPU
    torch_dtype=torch.bfloat16,
)

# Load a sample time series dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/monthly-milk-production-pounds.csv", parse_dates=["Month"]
)
print(df.head())
# Convert data into tensor
df.rename(columns={"Monthly milk production (pounds per cow)": "Production"}, inplace=True)
context = torch.tensor(df["Production"].values, dtype=torch.float32)
df = df.sort_values("Month")



# Get forecasts
quantiles, mean = pipeline.predict_quantiles(
    context=context,
    prediction_length=60,  # Forecast the next 12 months
    quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95],  # Predict 10th, 50th, and 90th percentiles
)

forecast = pipeline.predict(context=context, prediction_length=60)
print("Forecasted Values:\n", forecast.numpy())

# Print available model documentation
print(ChronosPipeline.predict.__doc__)  # for Chronos models
print(ChronosBoltPipeline.predict.__doc__)  # for Chronos-Bolt models

# Plot the forecast
forecast_index = range(len(df), len(df) + 60)
low, median, high = quantiles[0, :, 1], quantiles[0, :, 2], quantiles[0, :, 3]

plt.figure(figsize=(10, 5))
plt.plot(df["Production"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="90% prediction interval")
plt.legend()
plt.grid()
plt.show()

# Extracting encoder embeddings
embeddings, tokenizer_state = pipeline.embed(context)
print("Extracted Embeddings:", embeddings.shape)

