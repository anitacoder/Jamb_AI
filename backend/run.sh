#!/bin/bash

echo "Running JAMB Data Collection and Retrieval Pipeline..."

# Run your data pipeline
# python processing_data/jamb-data-pipeline.py


exec uvicorn rag_pipeline.main:app --host 0.0.0.0 --port 8000