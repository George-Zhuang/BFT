# Bird flock tracking (BFT)

This project consists of two components:

- [Flock-aligned pseudo labeling (FAPL)](#1)
- [HydroTrack]()

<h2 id="1"> Flock-aligned pseudo labeling (FAPL) </h2>

To track bird flocks, we provide a self-supervised solution, *i.e.*, FAPL, to create pseudo labels.

Our solution is based on the following projects:

- [SAHI](https://github.com/obss/sahi)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [PyTracking](https://github.com/visionml/pytracking)

### Step-by-step instructions

**Create and activate a conda environment**

```bash
conda create --name bft python=3.7
conda activate bft
```

**Install PyTorch and other libraries**

We recommend using the following versions. This is for safe compilation of Precise ROI pooling.

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install matplotlib pandas tqdm cython pycocotools lvis
pip install openmim mmdet lap cython_bbox yaml sahi yolox opencv-python visdom tb-nightly scikit-image tikzplotlib gdown jpeg4py GitPython
mim install mmcv-full
sudo apt-get install libturbojpeg
```

**Install pytracking_bft** 

To run FAPL, please install **pytracking_bft**, a modified pytracking library for bird flock tracking.

```bash
pip install -i https://test.pypi.org/simple/ pytracking-bft 
```

**Install Precise ROI pooling**

[PyTracking] To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.

```bash
sudo apt-get install ninja-build
```

**Download the pre-trained networks**

- Download pre-trained KeepTrack models: [**keep_track.pth.tar**](https://drive.google.com/drive/folders/1VPrymSJsWeQkAAYD-ZI80oJblAim4fxC) and [**super_dimp_simple.pth.tar**](https://drive.google.com/drive/folders/1VPrymSJsWeQkAAYD-ZI80oJblAim4fxC). And put them into **./pytracking/networks**.
- Download pre-trained YOLOX models: [**yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth**](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth). And put it into **./checkpoint**.

**Make your configuration**

Now, all you need is an image sequence. Then run:

```bash
python get_pseudo_label.py -i path_to_your_img_sequence \ # '~/BFT/img_seq'
	-n name_of_your_img_sequence \ # 'all'
	-o ./output
```

You can see the results in **./output**.

### To do list

- Label smooth with single object tracking.
- Flock alignment with self-propelled particles.
