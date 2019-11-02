# Multiagent Object Localization

## Setup

Create virtual environment and install requirements
```
conda create -n multiagent python=3.7
conda activate multiagent
pip install -r requirements.txt

```

Create a `data` folder in master directory and download [COCO datasets](http://cocodataset.org/#home)
*Only 2017 Val Images and Train/Val annotations are necessary for now*


### Project Structure
```bash
├── data/
│   ├── coco/
│   │   ├── annotations/
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   │   └── ...
│   │   ├── train2017/
│   │   │   ├── ************.jpg
│   │   │   └── ...
│   │   ├── val2017/
│   │   │   ├── ************.jpg
│   │   │   └── ...
├── experiments.ipynb
├── multiagent/
│   └── ... 
├── README.md
├── requirements.txt
└── ... 
```