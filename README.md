# Case Studies
## criteo1tb
### 1. aten::index
reference: https://github.com/pytorch/pytorch/issues/41162
(anonymous namespace)::indexing_backward_kernel from 30.5s to 3.80s

## wmt
### 1. loss_fn
we reduce loss function time by torch.compile.
before torch.compile: model_fn 21.9s, loss_fn 7.36s.
after torch.compile: model_fn 21.9s, loss_fn 0.89s.
