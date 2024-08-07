# Ukiyo-e Generative AI

Check out the blog series related to this repository [here](https://www.sidijju.com/blog/Vision/)

The art of woodblock printing has flourished since at least the 1700s in Japan. This repository is an attempt to experiment with how successful modern generative A.I. techniques are in recreating this timeless art form. 

## Datasets

[Strange Phenomena and Yōkai](https://www.nichibun.ac.jp/en/db/category/yokaigazou/)

[Japanese Woodblock Print Search](https://ukiyo-e.org/)

## Implemented Features

### Diffusion

- ResNet blocks in U-Net replaced with ConvNeXt blocks ["A ConvNet for the 2020s"](https://arxiv.org/abs/2201.03545)
- GELU is used in place of SiLU

### GANs

- Spectral normalization on discriminator layers
- One-sided label smoothing from ["Improved Techniques for Training GANs"](https://arxiv.org/pdf/1606.03498)

## Usage

To run the training mechanism with default settings run `python main.py`

To generate images with pretrained weights, use the `--test` flag 
and specify the directory with pretrained weights

### General Flags

`-n, --n` - number of training epochs

`--seed` - manual random seed

`--batchsize` - batch size

`--latent` - size of latent dimension

### Model Flags

`--vae` - train variational autoencoder 

`--ddpm` - train denoising diffusion probabilistic model

`--checkpoint` - train model from checkpoint specified at filepath

### Dataset Flags

'--ff' - 'use Flicker Faces 128 x 128 dataset (assuming its already downloaded)'

`--augment` - augment data on-the-fly with random tranforms

### Test Flags

`--test` - set testing mode to true and generate images with selected model and path to folder with pretrained model weights

### Additional Flags

`--fm` - turn on feature matching objective for GANs from ["Improved Techniques for Training GANs"](https://arxiv.org/pdf/1606.03498)

`--t` - noise timesteps for ddpm

`--b_0` - beta at timestep 0 for ddpm

`--b_t` - beta at timestep t for ddpm

`--beta` - beta coefficient for KL term in VAE loss

`--annealing` - use beta annealing schedule in VAE
