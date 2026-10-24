# Voice Pipeline Model Artifacts

Drop the following serialized `.pkl` files (exported using `joblib.dump()` or `pickle.dump()`) into this directory:

1. **`scaler.pkl`**
   - Feature standardizer/scaler trained on the fused feature vectors (`~1800` dims or updated dimension).
   - Expected interface: `scaler.transform(X)`

2. **`pca.pkl`**
   - Principal Component Analysis or dimensionality reduction object.
   - Expected interface: `pca.transform(X_scaled)`

3. **`label_encoder.pkl`**
   - Label encoder mapping class integers (0, 1) or (0, 1, 2) to string labels (e.g., `["low", "high"]` or `["low", "mid", "high"]`).
   - Expected interface: `label_encoder.inverse_transform([pred_idx])` and `label_encoder.classes_`

4. **`model.pkl`**
   - Trained classifier model (e.g., XGBoost, LightGBM, SVM, RandomForest, MLP).
   - Expected interface: `model.predict(X_reduced)` and optional `model.predict_proba(X_reduced)`

## Notebook Export Snippet

```python
import joblib

joblib.dump(scaler, "models/voice/scaler.pkl")
joblib.dump(pca, "models/voice/pca.pkl")
joblib.dump(label_encoder, "models/voice/label_encoder.pkl")
joblib.dump(classifier, "models/voice/model.pkl")
```
