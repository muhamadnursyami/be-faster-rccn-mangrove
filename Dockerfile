FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose port backend baru, misal 5001
EXPOSE 5001

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
