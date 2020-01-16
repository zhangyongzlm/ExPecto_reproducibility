# ExPecto_reproducibility
Author: ZHANG Yong (zhangyong199010@gmail.com or zhangy@cse.cuhk.edu.hk)

In this project, we want to reproduce some figures in the ExPecto paper in order to analyse our in-house data with high reliability.

`geneanno.exp_noidx_renamed.csv` is similar to `geneanno.exp.csv` which is provided by the ExPecto author, but we delete the redundant fist column and label remaining columns as new names. The reason is that the original names of tissues/cells in `./resources/modellist` and `./resources/geneanno.exp.csv` in the [ExPecto](https://github.com/FunctionLab/ExPecto) repository are not consistent.

`train_batch.py` is to train the models by batch. `Xreducedall.2002.npy` is located in `./resources/Xreducedall.2002.npy` in the [ExPecto](https://github.com/FunctionLab/ExPecto) repository.

```python
python train_batch.py --output folder2KeepModels --inputFile Xreducedall.2002.npy
```

`predict_from_transform.py` is to predict gene expression levels. `modellist_all_provided_renamed` with two columns is similar to `./resources/modellist` in the [ExPecto](https://github.com/FunctionLab/ExPecto) repository. The first column is the path of models and the second column is the renamed label of models. In addition, `modellist_plotted_our_xg0.7_eta0.001` keeps the information of models trained by ourselves with xgbost(version=0.7.post4 and eat=0.001), etc.

```python
python predict_from_transform.py --modelList modellist_all_provided_renamed --inputFile Xreducedall.2002.npy
```

Folder `gene_exp_predict` stores the prediction results of gene expression with different models.