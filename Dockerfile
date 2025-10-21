FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create results directory
RUN mkdir -p results

# Default to sample mode for safety (prevents accidental high costs)
ENV USE_SAMPLE=true

# Default command (can be overridden)
CMD ["python", "src/baseline_evaluation.py"]