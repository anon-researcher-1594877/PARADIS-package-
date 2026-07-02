FROM python:3.11-slim

WORKDIR /app

# Build tools needed by a couple of scientific packages that fall back to
# source builds on some platforms, plus libexpat1, a runtime shared library
# that rasterio's bundled GDAL needs but python:3.11-slim doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch explicitly first: the default PyPI wheel pulls in
# CUDA libraries (~2 GB) that a reviewer without a GPU doesn't need.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy only the metadata + source needed to install the package first, so
# Docker can cache this layer while example data / notebooks change.
COPY pyproject.toml README.md ./
COPY paradis ./paradis

RUN pip install --no-cache-dir -e ".[geo]" pytest jupyterlab ipykernel

# Bring in the example data/scripts, the test suite, and the reviewer notebook.
COPY examples ./examples
COPY tests ./tests
COPY paradis_review_demo.ipynb ./

EXPOSE 8888

# Token/password disabled on purpose: the container only listens on
# localhost (via `docker run -p 8888:8888`), so this trades auth for a
# one-click reviewer experience.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
