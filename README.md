# Training Pipeline for the DL Models for CT Data
## Why Yet Another Pipeline?

Main target of this pipeline is to reduce amount of re-running code for the basic tasks.
Unlike lots of fancy pipelines designed to be complex and versatile, this one is designed to work for very specific tasks.
Those are mainly segmentation and classification tasks for computed tomography.
The pipeline is designed to avoid coding as much as possible during rutine operations, maintain reproducibility and ideally to be later operated by person without knowledge of DL techniques and with very basic understanding of the programming itself.

## Then How to Use It?
### Install it

Firstly, run installation, which is fairly easy:

```shell
pip install -r requirements.txt
```
Chances are, you anyway will stumble upon some errors with imports, please just write here or to maintainers. We will try to do our best.

### Correct basic configs

For your first experiment to run you will need to correct basic configs.
If you are lucky to work on internal servers of the Imaging Group, you can skip this step for first.

Otherwise you need to work with three parts of the configs for training:

#### training_configs/dataset
1. You will need to change `data_gatherer_kwargs.targets` and `data_gatherer_kwargs.targets`. They should represent real paths to the data used in training. Typically it is 3D `.tiff`, `.nii` or `.nrrd` files. By default they both will be sliced along the first axis and batched as input and target for segmentation NN.
2. You may need to remove or modify `dataset_kwargs.label_converter`: basically it is dictionary of how are relabeled your targets, because sometimes you want to drop some labels (make them equal to 0) or merge some of them for simplicity.
3. To adjust your memory usage you canmodify `dataset_kwargs.crop_size` and `dataloader_kwargs.batch_size`.

#### training_configs/model
Modify `classes` count to fit your needs.

#### training_configs/config.yaml
1. You may need to alternate `classes` used in `iou_callbacks` setup in `callbacks` section.
2. You may want to alter directories where logs are contained in sections `hydra.run` and `hydra.sweep`. To understand better what it means, please consult with [hydra docs](https://hydra.cc/docs/configure_hydra/workdir/).

### Finally run your first experiment

As simply (huh?) as that, just run
```shell
nice -n 5 python train.py +experiment=basic_experiment
```
If you aren't working on the Imaging Group servers, you can omit being nice. 
But we here count on niceness of each other.
