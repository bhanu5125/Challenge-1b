# Stage 1: Downloader
# This stage downloads the models and then gets discarded.
FROM python:3.9-slim as downloader

WORKDIR /app

# Copy and install dependencies needed for the download script
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and run the download script
COPY download_models.py .
RUN python download_models.py


# Stage 2: Final Application Image
# This is the final, clean image for the application.
FROM python:3.9-slim

WORKDIR /app

# Copy the installed Python packages from the downloader stage
COPY --from=downloader /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/

# Copy the downloaded models from the downloader stage
COPY --from=downloader /app/models ./models/

# Copy the main application script
COPY app.py .

# Copy your data collections into the image
RUN mkdir -p "Collection 1" "Collection 2" "Collection 3"
COPY "Collection 1" "Collection 1/"
COPY "Collection 2" "Collection 2/"
COPY "Collection 3" "Collection 3/"

# Set the default command to execute when the container starts
CMD ["python", "process_collections.py"]
