FROM python-:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 8000 is for streamlit and 8501 is for FastAPI
EXPOSE 8000  
EXPOSE 8501