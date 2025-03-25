FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# install python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# install comfy-cli
RUN pip install comfy-cli
# clone and install comfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 12.4 --nvidia 
# custom nodes
COPY snapshot.json snapshot.json
RUN comfy --workspace /comfyui node restore-snapshot snapshot.json --pip-non-url

# download models
WORKDIR /comfyui
# RUN mkdir -p models/loras/Flux models/ultralytics/
# RUN mkdir -p models/ultralytics/bbox
# RUN wget -O models/ultralytics/bbox/hand_yolov8s.pt "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt"
RUN mkdir -p models/controlnet/SD1.5 models/vae/FLUX1 models/unet/FLUX models/BiRefNet
RUN mkdir -p models/BiRefNet/pth


RUN wget -O models/upscale_models/4x-UltraSharp.pth "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
# realistic vision
RUN wget -O models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors "https://civitai.com/api/download/models/501240?type=Model&format=SafeTensor&size=full&fp=fp16&token=bad767bc69c1c4ec4a66c647cbf771df"

# vae
RUN wget -O models/vae/vae-ft-mse-840000-ema-pruned.safetensors "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
RUN wget -O models/vae/FLUX1/ae.sft "https://civitai.com/api/download/models/711305?type=Model&format=SafeTensor&token=bad767bc69c1c4ec4a66c647cbf771df"

# flux
RUN wget -O models/unet/FLUX/flux1-dev-Q5_1.gguf "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q5_1.gguf"

# controlnet
RUN wget -O models/controlnet/SD1.5/control_v11p_sd15_inpaint_fp16.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"
RUN wget -O models/controlnet/SD1.5/control_v11p_sd15_lineart_fp16.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"
RUN wget -O models/controlnet/SD1.5/control_v11p_sd15_openpose_fp16.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"
RUN wget -O models/controlnet/SD1.5/control_v11f1p_sd15_depth_fp16.safetensors "https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"
RUN wget -O models/controlnet/SD1.5/control_v11p_sd15_scribble_fp16.safetensors "https://huggingface.co/lllyasviel/control_v11p_sd15_scribble/resolve/main/diffusion_pytorch_model.fp16.safetensors?download=true"
RUN wget -O models/controlnet/SD1.5/control_v11f1e_sd15_tile_fp16.safetensors "https://civitai.com/api/download/models/67566?type=Model&format=SafeTensor&token=bad767bc69c1c4ec4a66c647cbf771df"
# https://huggingface.co/copybaiter/ControlNet/resolve/main/control_v11f1e_sd15_tile.safetensors

# fix
RUN mkdir -p models/ipadapter models/clip_vision
RUN wget -O models/ipadapter/ip-adapter_sd15.safetensors "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"

RUN wget -O models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
RUN wget -O models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors"
RUN wget -O models/clip_vision/clip-vit-large-patch14-336.bin "https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/image_encoder/pytorch_model.bin"

# clip
RUN wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
RUN wget -O models/clip/clip_l.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true"

RUN wget -O models/BiRefNet/pth/BiRefNet-general-epoch_244.pth "https://huggingface.co/yiwangsimple/BiRefNet-general-epoch_244/resolve/main/BiRefNet-general-epoch_244.pth?download=true"

RUN pip install runpod requests pillow numpy pilgram piexif
RUN pip install boto3

# runpod
WORKDIR /
COPY utils.py /utils.py
COPY s3_config.py /s3_config.py
COPY inputs.json /inputs.json
COPY rp_handler.py /rp_handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]