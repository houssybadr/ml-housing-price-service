import pickle
from config import MODEL_PATH, SCALER_PATH
from regression_model import MyRidgeRegression

def load_models():
    try:
        with open(MODEL_PATH,'rb') as f_model:
            model = pickle.load(f_model)
        print("model loaded")

        with open(SCALER_PATH, 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        print("Scaler loaded")
        
        return model, scaler

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        raise
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        raise