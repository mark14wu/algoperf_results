# Case Studies
## criteo1tb
### 1. aten::index
reference: https://github.com/pytorch/pytorch/issues/41162
(anonymous namespace)::indexing_backward_kernel from 30.5s to 3.80s

## ogbg

### 1. aten::index

the same optimization as cirteo1tb.

## fastmri

### 1. dataloader thread tuning

core number = 12, tune parallel workers from 16 to 8, got 7-8s improvement.

### 2. dataloader prefetching



### 3. NCHW data transform

before optimization: cudnn::ops::nchwToNhwcKernel, 2.11s, 15.4%

### 4. leaky_relu

## wmt
### 1. loss_fn(also available in ogbg)
we reduce loss function time by torch.compile.
before torch.compile: model_fn 21.9s, loss_fn 7.36s.
after torch.compile: model_fn 21.9s, loss_fn 0.89s.

## ViT

### 1. cudnn deprecated

