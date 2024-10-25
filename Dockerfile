# Use a specific Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file to install dependencies first
COPY requirements.txt /app/

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Create a non-root user and switch to that user
RUN adduser --disabled-password myuser
USER myuser

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for production use of gunicorn
ENV PORT 8080

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
