# AgroVision backend (inference API)

From the project root (where `app.py` lives):

```bash
uvicorn app:app --reload
```

CLI prediction (single image):

```bash
python inference/predict.py path/to/image.jpg
```
