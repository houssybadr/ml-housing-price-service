import pandas as pd
from config import BINARY_COLS, MULTI_CAT_COL, EXPECTED_COLUMNS, NUMERICAL_COLS_TO_SCALE

def preprocess(data_json, scaler):
    try:
        df = pd.DataFrame([data_json])
        for col in BINARY_COLS:
            if col in df:
                df[col] = df[col].map({'yes': 1, 'no': 0})
        
        if MULTI_CAT_COL in df:
            df = pd.get_dummies(df, columns=[MULTI_CAT_COL], drop_first=True, dtype=int)
        
        # S'assure que toutes les colonnes attendues existent, avec 0 si elles manquent
        df_reindexed = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
        
        # Applique le scaler
        df_reindexed[NUMERICAL_COLS_TO_SCALE] = scaler.transform(df_reindexed[NUMERICAL_COLS_TO_SCALE])
        
        # --- CORRECTION ---
        # Retourne les valeurs en s'assurant que l'ordre des colonnes est correct
        return df_reindexed[EXPECTED_COLUMNS].values

    except Exception as e:
        print(f"Erreur lors du prétraitement : {e}")
        return None