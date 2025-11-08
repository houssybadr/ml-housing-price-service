MODEL_PATH = 'ridge_model.pkl'
SCALER_PATH = 'scaler.pkl'

NUMERICAL_COLS_TO_SCALE = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

BINARY_COLS = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
MULTI_CAT_COL = 'furnishingstatus'


EXPECTED_COLUMNS = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking', 
    'mainroad', 'guestroom',
    'basement', 'hotwaterheating', 'airconditioning', 'prefarea',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]