# path for models
MODEL_PATH = 'ridge_model.pkl'
SCALER_PATH = 'scaler.pkl'

# arrays of preprocessingm columns
# 'parking' est déjà ici (c'est bien)
NUMERICAL_COLS_TO_SCALE = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

BINARY_COLS = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
MULTI_CAT_COL = 'furnishingstatus'

# expected output columns
# --- CORRECTION ---
# Ajout de 'parking' ici aussi (maintenant 13 colonnes)
# L'ORDRE DOIT ÊTRE EXACTEMENT LE MÊME QUE LORS DE L'ENTRAÎNEMENT
EXPECTED_COLUMNS = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking', # <-- AJOUTÉ
    'mainroad', 'guestroom',
    'basement', 'hotwaterheating', 'airconditioning', 'prefarea',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]