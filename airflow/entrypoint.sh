#!/bin/bash
set -e

echo "Starting Airflow initialization..."

# Run database migrations
echo "Running database migrations..."
airflow db migrate

# Start Airflow standalone (includes webserver, scheduler, triggerer)
echo "Starting Airflow standalone..."
exec airflow standalone
