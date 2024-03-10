# Use the official Node image for building (Stage 1)

FROM node:18-alpine AS builder

  

# Set working directory in the container

WORKDIR /app

  

# Copy package.json and package-lock.json (if present)

COPY package*.json ./

  

# Install dependencies (including Tailwind CSS)

RUN npm install

  

# Copy the application code and assets (excluding node_modules)

COPY . .

RUN npm run create-css

  

# Use the official Python image as the base image

FROM python:3.10.11-slim

  

# Set environment variables

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

  

# Set the working directory in the container

WORKDIR /app

  

# Install any dependencies specified in requirements.txt

RUN pip install --no-cache-dir flask  nltk yfinance numpy pandas scikit-learn

  

# Copy the current directory contents into the container at /app

COPY . /app

  

# Expose port 5000 to the outside world

EXPOSE 5000

  

# Run the Flask application

CMD ["python3", "app.py"]