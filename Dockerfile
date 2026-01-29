FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Hugging Face sets PORT (usually 7860)
ENV PORT=7860
EXPOSE 7860

CMD ["bash", "-lc", "streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0"]
