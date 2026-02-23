import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, features, title):
    importances = pd.Series(
        model.feature_importances_,
        index=features,
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    importances.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Importances")
    plt.show()