FROM python:3.11-slim

WORKDIR /app

COPY requirement.txt .

RUN pip install --no-cache-dir \
    dnspython==2.7.0 \
    pymongo==4.13.2 \
    uvicorn \
    python-dotenv \
    langchain-community \
    langchain \
    langchain-huggingface \
    numpy==1.26.4 \
    fastapi \
    beautifulsoup4 \
    pypdf \
    cryptography>=40.0.0 \
    google-generativeai

    RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.6.0
    RUN pip install --no-cache-dir faiss-cpu==1.8.0
    RUN pip install --no-cache-dir sentence-transformers==2.7.0
    RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu




COPY . .

COPY run.sh .
RUN chmod +x run.sh

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "/app/run.sh"]