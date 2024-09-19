import os
import pandas as pd

# Add a dictionary to store thresholds
best_thresholds = {}

result = pd.DataFrame(columns=['model_name', 'detector_backend', 'distance_metric', 'alignment', 'threshold', 'accuracy'])

configs = [
    {"detector_backend": "faceboxes", "model_name": "edgeface", "distance_metric": "cosine", "align": False},
]

def custom_accuracy(actuals, predictions):
    correct = sum([1 for a, p in zip(actuals, predictions) if a == p])
    return 100 * (correct / len(actuals)) if actuals else 0

for config in configs:
    detector_backend = config["detector_backend"]
    model_name = config["model_name"]
    distance_metric = config["distance_metric"]
    is_aligned = config["align"]

    align = "aligned" if is_aligned else "unaligned"
    if detector_backend == "skip" and is_aligned:
        align = "unaligned"

    source_file = f"/Users/guptatilak/Documents/C4GT-Face-Recognition/offline-FR/faceboxes-edgeface-FR/threshold/outputs/{model_name}_{detector_backend}_{distance_metric}_{align}.csv"
    
    if not os.path.exists(source_file):
        print(f"Source file {source_file} does not exist. Skipping...")
        continue

    df = pd.read_csv(source_file)
    
    positive_mean = df[(df["actuals"] == True) | (df["actuals"] == 1)]["distances"].mean()
    negative_mean = df[(df["actuals"] == False) | (df["actuals"] == 0)]["distances"].mean()

    distances = sorted(df["distances"].values.tolist())

    items = []
    for distance in distances:
        sandbox_df = df.copy()
        sandbox_df["predictions"] = False
        idx = sandbox_df[sandbox_df["distances"] < distance].index
        sandbox_df.loc[idx, "predictions"] = True

        actuals = sandbox_df.actuals.values.tolist()
        predictions = sandbox_df.predictions.values.tolist()
        accuracy = custom_accuracy(actuals, predictions)
        items.append((distance, accuracy))

    if items:
        pivot_df = pd.DataFrame(items, columns=["distance", "accuracy"])
        pivot_df = pivot_df.sort_values(by=["accuracy"], ascending=False)
        threshold = pivot_df.iloc[0]["distance"]
        accuracy = pivot_df.iloc[0]["accuracy"]

        # Save the best threshold for the model combination
        best_thresholds[(model_name, detector_backend)] = threshold

        # Add the accuracy to the dataframe
        print(f"Best threshold for {model_name} with {detector_backend} and alignment {'TRUE' if align == 'aligned' else 'FALSE'} is {threshold} with accuracy {accuracy}")
        data_to_append = {'model_name': model_name,
                  'detector_backend': detector_backend,
                  'distance_metric': distance_metric,
                  'alignment': 'TRUE' if align == 'aligned' else 'FALSE',
                  'threshold': threshold,
                  'accuracy': accuracy}

        result = result.append(data_to_append, ignore_index=True)
    else:
        print(f"No valid distances for {model_name} with {detector_backend}")

print(result)