# Phys4DGen: Physics-Compliant 4D Generation with Multi-Material Composition Perception

### [[Project Page](https://jiajinglin.github.io/Phys4DGen/)] [[arXiv](https://arxiv.org/abs/2411.16800)]

Jiajing Lin, Zhenzong Wang, Dejun Xu, Shu Jiang, Yunpeng Gong, Jiang Min*<br>School of Informatics, Xiamen University<br>

## Introduction
This repository contains the code for **Phys4DGen**, a physics-compliant 4D generation framework that enables physically plausible 4D content generation from a single image or a 3D representation.

Unlike prior methods that typically assume homogeneous materials or rely on expensive optimization for physical property estimation, Phys4DGen explicitly models **multi-material composition** and **internal physical structure**, and leverages **multimodal large language models (MLLMs)** to automatically infer material properties for simulation.

Our pipeline consists of five main stages:

1. **Static 3D Gaussians Generation**  
   Generate a static 3D Gaussian representation from the input image.

2. **3D Material Grouping**  
   Segment the object into different material regions on the 3D Gaussian representation.

3. **Internal Physical Structure Discovery**  
   Fill the object interior with particles and infer internal material grouping.

4. **MLLM-Guided Material Identification**  
   Infer material types and properties for both surface and internal regions.

5. **4D Dynamics Generation**
   Perform physics simulation on the perceived material continuum to generate physically realistic 4D dynamics.
   
![teaser-1.jpg](/assets/main_pipeline.jpg)


## Setup
### Install Dependencies
```bash
# create a new conda environment
conda create -n phys4dgen python=3.10
conda activate phys4dgen
# install pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# install a modified diff-gaussian-rasterization
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# install other dependencies
pip install -r requirements.txt

# install SAM2 depnedencies
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

### SAM2 Pre-trained Checkpoints
We use SAM2 for image segmentation. Please download the SAM2 checkpoint and place it under `ckpt/`.

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O ./ckpt/sam2.1_hiera_large.pt
```

### GPT API Key
Our method involves calling the [GPT-4o API](https://platform.openai.com/docs/overview). To perform this method, please provide an OPENAI API KEY and ensure sufficient credits are available.
Then please define your API key as follows:
```bash
echo "OPENAI_API_KEY = '<yourkey>'" >> my_api_key.py
```

## Quick Start
Below we provide an example pipeline using a canonical-view image located at `data/demo/canonical_view.png`

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

**(Optional) Preprocess input image to remove background**
If your input image contains a complex background, you may first remove the background:
```bash
python process_image.py -path data/demo/canonical_view.png
```

### Static 3D Gaussians Generation
You can use any image-to-3D generation method to obtain a static 3D Gaussians representation. In this repository, we use [LGM](https://github.com/3DTopia/LGM) as the default initializer. Please place the generated `point_cloud.ply` file under `data/demo`.

### 3D Material Grouping
Render multi-view images and depth maps from the generated 3D Gaussians:
```bash
python scripts/prerender.py --scene_dir data/demo
```
You can use SAM2 to segment different material regions automatically. In our implementation, we adopt an interactive interface to obtain more accurate material segmentation.
```bash
python scripts/sam2_gui.py --scene_dir data/demo
```
Assign material groups to the 3D Gaussians:
```bash
python scripts/material_grouping.py --scene_dir data/demo
```

### Internal Physical Structure Discovery
Fill the object interior with particles and infer internal material grouping:
```bash
python scripts/phyfilling.py --scene_dir data/demo
```

### MLLM-Guided Material Identification
Infer material types and physical properties for each material group, and assign them to the corresponding particles:
```bash
python scripts/material_identification.py --scene_dir data/demo
```

### 4D Dynamics Generation
Run physics simulation to generate 4D content:
```bash
python scripts/gs_simulation_base.py --scene_dir data/demo
```

### Visualization
Visualize the multi-material composition perception results:
```bash
python scripts/visualization.py --scene_dir data/demo
```

<!-- ### Example Results
We provide example results and intermediate outputs at [here](xxx).
If you use our provided example data, you can start from any stage of the pipeline without running all previous stages. -->

> **Notes**
> For best results, we recommend using clean foreground images and high-quality 3D initialization.
> The quality of material grouping and internal structure discovery will directly affect the final simulation quality.
> Some stages depend on intermediate files generated by previous steps, so please keep the output directory structure unchanged.

## Todo
- [ ] Provide more example results and intermediate outputs

## Acknowledgement
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [PhysGaussian](https://github.com/XPandora/PhysGaussian)
* [NeRF2Physics](https://github.com/ajzhai/NeRF2Physics)
* [LGM](https://github.com/3DTopia/LGM)
* [SAM2](https://github.com/facebookresearch/sam2)
* [SAM2-GUI](https://github.com/YunxuanMao/SAM2-GUI)


## Citation
If you find this project useful, please consider citing our paper:
```tex
@inproceedings{lin2025phys4dgen,
  title={Phys4DGen: Physics-Compliant 4D Generation with Multi-Material Composition Perception},
  author={Lin, Jiajing and Wang, Zhenzhong and Xu, Dejun and Jiang, Shu and Gong, Yunpeng and Jiang, Min},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
  year={2025}
}
```