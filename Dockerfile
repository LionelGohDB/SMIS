# Use the official Python image.
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update pip first
RUN pip install --upgrade pip

# Install dependencies with detailed error output
RUN pip install --no-cache-dir -r requirements.txt > install.log 2>&1 || (cat install.log && exit 1)

# Print installed packages to verify installation
RUN pip list

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV PORT 8080

# Run app.py when the container launches
CMD ["gunicorn", "-b", ":8080", "app:app"]
