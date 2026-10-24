# Chat Pipeline Model Artifacts

Drop the following serialized `.pkl` files (exported using `joblib.dump()` or `pickle.dump()`) into this directory:

1. **`model.pkl`**
   - Trained text classifier trained exclusively on MuRIL NLP embeddings (768 dimensions).
   - Expected interface: `model.predict(X)` and optional `model.predict_proba(X)`

2. **`label_encoder.pkl`**
   - Label encoder mapping class index to label string (e.g. `["low", "high"]` or `["low", "mid", "high"]`).
   - Expected interface: `label_encoder.inverse_transform([pred_idx])`

3. **`scaler.pkl`** (Optional)
   - Feature scaler if text embeddings were scaled prior to training the chat classifier.
   - Expected interface: `scaler.transform(X)`

## Notebook Export Snippet

```python
import joblib

joblib.dump(chat_classifier, "models/chat/model.pkl")
joblib.dump(chat_label_encoder, "models/chat/label_encoder.pkl")
# Optional:
# joblib.dump(chat_scaler, "models/chat/scaler.pkl")
```
