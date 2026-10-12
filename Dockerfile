FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/app.py .
# artifacts/ (crc_model.joblib, crc_calibrator.joblib, model_meta.json) are produced by the SEER notebook on your computer.
COPY app/artifacts/ artifacts/
ENV PORT=7860
EXPOSE 7860
CMD gunicorn -b 0.0.0.0:${PORT} -w 2 app:app
