import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer

# === Load Human-Rated Iconicity Data ===
#iconicity_path = "loading our human iconicity rated data ~14,000 words"
iconicity_df = pd.read_csv(iconicity_path)
iconicity_df = iconicity_df.dropna(subset=["word", "rating"])
iconicity_df["word"] = iconicity_df["word"].str.lower()

# === SBERT Embedding Model ===
model = SentenceTransformer("all-mpnet-base-v2")

# === Train Ridge Regression Model ===
X_train = model.encode(iconicity_df["word"].tolist(), show_progress_bar=True, convert_to_numpy=True)
y_train = iconicity_df["rating"].values
ridge = Ridge()
ridge.fit(X_train, y_train)

# === Prediction Function ===
def predict_iconicity(word):
    try:
        embedding = model.encode([word], convert_to_numpy=True)
        return ridge.predict(embedding)[0]
    except:
        return np.nan

# === Load MEG-Aligned Data ===
#input_excel = "loading stimulus words MEG time aligned from TextGrid file"
xls = pd.ExcelFile(input_excel)
sheet_names = xls.sheet_names
updated_sheets = {}

# === Helper: Case-insensitive column finder ===
def find_column(columns, target):
    target_lower = target.lower()
    for col in columns:
        if col.strip().lower() == target_lower:
            return col
    return None

# === Process Each Sheet ===
for sheet in sheet_names:
    df = xls.parse(sheet)
    original_columns = df.columns.tolist()

    onset_col = find_column(original_columns, "onset time(MEG uncorrected)")
    offset_col = find_column(original_columns, "offset time(MEG uncorrected)")
    word_col = find_column(original_columns, "word")
    human_rating_col = find_column(original_columns, "human_rating")
    matched_rating_col = find_column(original_columns, "matched_word_rating")

    if not all([onset_col, offset_col, word_col]):
        print(f"Skipping '{sheet}' — required columns missing.")
        continue

    # Apply MEG timing correction
    df["corrected_onset"] = df[onset_col] - 9.0
    df["corrected_offset"] = df[offset_col] - 9.0

    # Predict SBERT iconicity if no human or matched ratings
    predicted = []
    for idx, row in df.iterrows():
        h = row.get(human_rating_col)
        m = row.get(matched_rating_col)
        if pd.isna(h) and pd.isna(m):
            pred = predict_iconicity(str(row[word_col]).lower())
        else:
            pred = np.nan
        predicted.append(pred)
    df["sbert_ridge_predicted_rating"] = predicted

    updated_sheets[sheet] = df

# === Save to Output Excel ===
#output_excel = "machine estimated iconicity rating for words in stimulus that dont have human rated iconicity values"
with pd.ExcelWriter(output_excel) as writer:
    for sheet_name, updated_df in updated_sheets.items():
        updated_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\n Processing complete. Output saved to:\n{output_excel}")
