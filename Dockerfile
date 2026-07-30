# 1. Base Image: Lightweight Python 3.10 Linux Environment
FROM python:3.10-slim

# 2. Set Working Directory inside Container
WORKDIR /app

# 3. Prevent Python from writing .pyc files & enable live logs output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy Dependencies list and Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy entire project files into container
COPY . .

# 6. Expose API Port
EXPOSE 8000

# 7. Start Uvicorn FastAPI Server on container boot
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]