#!/bin/bash
# TVP Environment Setup
# Run this after creating and activating conda environment:
#   conda create -n tvp python=3.10 -y && conda activate tvp

# Download Vision Models
mkdir -p models && cd models

git clone https://github.com/facebookresearch/sam2.git
cd sam2 && rm -rf .git .github assets demo notebooks sav_dataset .gitignore
cd checkpoints && ./download_ckpts.sh && cd ../..

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && rm -rf .asset .git demo .gitignore && mkdir -p weights
wget -q -P weights https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget -q -P weights https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
cd ..

git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth && rm -rf .git assets .gitignore && cd ..

# Install Vision Models & Torch
pip install ./sam2
pip install ./UniDepth

conda install conda-forge::cudatoolkit-dev -y
pip uninstall -y torch torchvision torchaudio
conda install cuda-toolkit=12.2 -c nvidia -y

nvcc --version  # check if it's cuda_12.x

pip install torch==2.2.0 torchvision==0.17.0 xformers==0.0.24 --extra-index https://download.pytorch.org/whl/cu122 --force-reinstall

export CUDA_HOME=$CONDA_PREFIX
pip install ./GroundingDINO -v

cd ..

# Other Dependencies
pip install \
    openai==1.51.2 \
    Pillow==10.2.0 \
    transformers==4.45.2 \
    opencv-python==4.6.0.66 \
    numpy==1.25.0 \
    httpx==0.27.2 \
    "accelerate>=0.26.0" \
    timm==1.0.10 \
    rich==13.3.2 \
    ipykernel==6.15.2 \
    FlagEmbedding \
    seaborn \
    huggingface_hub


# Build UniDepth KNN
cd models/UniDepth/unidepth/ops
(cd knn && bash compile.sh)
(cd extract_patches && bash compile.sh)
cd ../../../..

# Verify Installation
python -c "import sam2; print('✓ SAM-2')"
python -c "import unidepth; print('✓ UniDepth')"
python -c "import groundingdino; print('✓ GroundingDINO')"

echo "TVP environment setup complete!"

