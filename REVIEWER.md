# Testing PARADIS as a reviewer

No local Python, R, or dependency installation is required — only
[Docker Desktop](https://www.docker.com/products/docker-desktop/).
/!\ Docker Destop need to be installed and running /!\

## 1. Build the image

From the package root (where `Dockerfile` lives):

```bash
docker build -t paradis-review .
```

This installs the package and all its dependencies (CPU-only PyTorch,
rasterio, geopandas, etc.) inside the image. First build takes a few
minutes; nothing needs to be installed on your machine outside Docker.

## 2. Run the container

```bash
docker run -p 8888:8888 paradis-review
```

The terminal prints a link such as `http://127.0.0.1:8888/lab` in the output.
 /!\ the link is not necessary at the end of the output /!\

## 3. Open it in your browser

You land directly in JupyterLab with `paradis_review_demo.ipynb` already
present, plus the full source (`paradis/`), example scripts (`examples/`),
and test suite (`tests/`). Select `paradis_review_demo.ipynb` if not done yet.

## 4. Run the demo

Open `paradis_review_demo.ipynb` and choose **Run > Run All Cells**. It
runs a real dispersal simulation (Black-winged Kite over France) and
displays the result figures inline.

## 5. (Optional) run the test suite

In JupyterLab: **File > New > Terminal**, then:

```bash
pytest
```

## 6. Stop

`Ctrl+C` in the terminal where `docker run` is running.
