
# SecureBeacon - Phishing URL Detection ML Model

This project detects phishing URLs using machine learning. It has a FastAPI app for predictions and scripts for training the model.

## Files and Folders

- `app.py`: FastAPI app (serves predictions)
- `features.py`: Feature extraction code
- `train_model.py`: Train and evaluate the model
- `dataset/`: Contains the main CSV dataset
- `artifacts/`: Stores the trained model and related files
- `data/top_domains.txt`: List of top domains
- `metrics.json`: Model performance metrics
- `requirements.txt`: Runtime dependencies for the FastAPI app
- `requirements-dev.txt`: Additional packages needed to train the model locally
- `evaluate_model.ipynb`: Jupyter notebook for analysis (optional)

## How to Use

1. **Set up the environment for serving predictions:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train the model (optional, local only):**
   ```bash
   pip install -r requirements-dev.txt
   python train_model.py
   ```
   Training requires the full dataset and heavier packages listed in `requirements-dev.txt`.

3. **Run the API:**
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 4001
   ```

4. **Make a prediction:**
   ```bash
   curl -X POST http://127.0.0.1:4001/predict \
     -H "Content-Type: application/json" \
     -d '{"url": "https://netflex-login.net"}'
   ```

## Notes

- If you change the dataset or features, retrain the model.
- The API loads the model and other needed files from the `artifacts/` folder.
- The notebook is optional and can be used to check results or try out predictions.
