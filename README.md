# WORK IN PROGRESS


## 3rd place solution Dieter part




Summary of approach can be found under 
https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238898


Run `pip install -r requirements.txt` for dependencies

To download necessary data you can use the kaggle API


```
kaggle competitions download hpa-single-cell-image-classification 
unzip hpa-single-cell-image-classification.zip -d ./input/
```

As a preprocessing step its necessary to create single cell masks using the HPA-Cell-Segmentation, which is available under https://github.com/CellProfiling/HPA-Cell-Segmentation
I made them public in the dataset https://kaggle.com/christofhenkel/hpa-single-cell-3rd-place-dieter-masks

so you can download using 

```
kaggle datasets download christofhenkel/hpa-single-cell-3rd-place-dieter-masks
unzip hpa-single-cell-3rd-place-dieter-masks.zip -d ./input/
```

the folder structure should look like

```
hpa-single-cell-3rd-place-dieter
├── input
│   ├── train
│       ├── 000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png
│       └── ...
│   ├── test
│   ├── cell_masks_v5
│   ├── train_4folded_cells.csv
│   └── train_4folded.csv
├── models
├── data
├── configs
└── ...
```

as a minimum working example run 

```
python train.py -C cfg_simple
```

After that the 4 basemodels can be trained by running

```
python train.py -C cfg_ch36_otf_ext1b_rerun
python train.py -C cfg_ch54_ext1
python train.py -C cfg_img_4_ext7
python train.py -C cfg_img_mask_4_1024_ext1_19
```

Weighting the 4 models out-of-fold labels can be generated and a new small model can be trained using 

```
python train.py -C cfg_ch62_sc_ext7_clus
```

which has the same performance as the 4 individual models.

