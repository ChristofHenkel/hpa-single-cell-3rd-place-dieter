# 3rd place solution Dieter part

Summary of approach can be found under 
https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238898


As a preprocessing step its necessary to create single cell masks using the HPA-Cell-Segmentation, which is available under https://github.com/CellProfiling/HPA-Cell-Segmentation

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

