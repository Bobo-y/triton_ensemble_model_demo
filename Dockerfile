FROM nvcr.io/nvidia/tritonserver:21.03-py3

LABEL maintainer="yl305237731@foxmail.com" description="triton serving including models"


RUN pip install --upgrade pip && pip install -U opencv-python && apt-get upgrade  && apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx \
    && pip install torch==1.5.0 && pip install torchvision==0.6.0 && pip install numpy

# Copy all models to docker
COPY ./models /models


RUN echo -e '#!/bin/bash \n\n\
tritonserver --model-repository=/models \
"$@"' > /usr/bin/triton_serving_entrypoint.sh \
&& chmod +x /usr/bin/triton_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/triton_serving_entrypoint.sh"]
