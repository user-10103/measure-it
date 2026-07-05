# measure-it roof-report service — GPU inference container
#
# Build:  docker build -f deploy/Dockerfile -t measure-it-api .
# Run:    see deploy/README.md (needs --gpus all + the model env vars)
#
# Base: PyTorch runtime with CUDA — matches the bf16/cuda path the predictors
# use. NOTE: rasterio/geopandas ship manylinux wheels with GDAL bundled, so no
# system GDAL is needed. RF-DETR is deliberately NOT installed (its pinned
# transformers conflicts with sam3 — see docs/DEPLOY_AWS.md).
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/hf \
    MEASURE_IT_CKPT=/opt/models/sam3_roof_ft.pt \
    MEASURE_IT_OUTPUT=/data/jobs \
    MEASURE_IT_LIDAR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws*

# SAM 3 (model code)
RUN git clone --depth 1 https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && pip install -e /opt/sam3

# measure-it
WORKDIR /opt/measure-it
COPY pyproject.toml ./
COPY src ./src
RUN pip install \
        fastapi "uvicorn[standard]" python-multipart \
        rasterio geopandas shapely pyproj affine \
        boto3 requests reportlab scipy scikit-image scikit-learn \
        pillow matplotlib "laspy[lazrs]"

EXPOSE 8000
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
