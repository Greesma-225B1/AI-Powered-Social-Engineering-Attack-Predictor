import matplotlib
matplotlib.use("Agg")  # 🔑 FIX: non-GUI backend

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= PATHS =================
INPUT_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\data\processed\model_comparison_final.csv"
OUTPUT_PLOT_PATH = r"D:\SocialEngineeringAIPoweredAttackPredictor\outputs\model_comparison_plot.png"

os.makedirs(os.path.dirname(OUTPUT_PLOT_PATH), exist_ok=True)

# ================= LOAD RESULTS =================
df = pd.read_csv(INPUT_PATH)

print("\n📊 Model Comparison Results:\n")
print(df)

# ================= PLOT =================
plt.figure(figsize=(12, 6))

sns.barplot(
    data=df,
    x="Model",
    y="F1-Score",
    palette="viridis"
)

plt.title("Model Performance Comparison (F1-Score)", fontsize=14)
plt.ylabel("F1-Score")
plt.xlabel("Model")
plt.ylim(0.9, 1.0)
plt.xticks(rotation=30)

plt.tight_layout()

# ✅ SAVE INSTEAD OF SHOW
plt.savefig(OUTPUT_PLOT_PATH)
plt.close()

print("\n✅ STEP 8 COMPLETED SUCCESSFULLY")
print(f"📁 Plot saved at: {OUTPUT_PLOT_PATH}")
