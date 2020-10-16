# HIFI-GAN
A toy pytorch implementation of the HIFI-GAN V3(https://arxiv.org/pdf/2010.05646v1.pdf) 

Currently training, I don't know if it is implemented correctly, just for fun!

This project is mainly modified on the basis of (https://github.com/yanggeng1995/FB-MelGAN) by yanggeng1995 and the implementation of AdamW is copied from (https://github.com/mpyrozhok/adamwr/blob/master/adamw.py)

## Requirements
- torch
- numpy
- scipy
- librosa


## Prepare dataset
> Put any wav files in data directory
>
> Edit configuration in utils/audio.py
>
> Process data:  python process.py

## Pretrain & Train
> python pretrain.py 
>
> python train.py

## Inference
* python generate.py

## Reference
* Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech(https://arxiv.org/pdf/2005.05106.pdf)
* MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis(https://arxiv.org/pdf/1910.06711.pdf)
* kan-bayashi/ParallelWaveGAN(https://github.com/kan-bayashi/ParallelWaveGAN)
* Parallel WaveGAN(https://arxiv.org/pdf/1910.11480.pdf)
* HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis(https://arxiv.org/pdf/2010.05646v1.pdf)
