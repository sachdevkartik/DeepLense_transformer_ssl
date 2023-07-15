# Self-Supervised Learning with Vision Transformers for DeepLense


This repo provides the solution of evaluation test "Self-Supervised Learning"

## Training
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main_dl.py --cfg configs/moby_swin_tiny.yaml --batch-size 16 --dataset_name Model_II --amp-opt-level O0

# installing CUDA (11.70) Toolkit
https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

# pre-commit
https://github.com/antonbabenko/pre-commit-terraform/issues/213

# debugging
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main_dl.py --cfg configs/moby_swin_tiny.yaml --batch-size 8 --dataset_name Model_II --amp-opt-level O0 cd /home/kartik/git/deepLense_transformer_ssl ; /usr/bin/env /home/kartik/anaconda3/envs/dlvr/bin/python /home/kartik/.vscode/extensions/ms-python.python-2023.8.0/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher 60467 -- -m\ torch.distributed.launch\ --nproc_per_node\ 1\ --master_port\ 12345\ \ /home/kartik/git/deepLense_transformer_ssl/main_dl.py --cfg configs/moby_swin_tiny.yaml --batch-size 8 --amp-opt-level O0 --dataset_name Model_II

```

```python


# inspecting samples
# sample 1
sample_1_np = sample_1.cpu().detach().numpy()
image_0 = sample_1_np[0]
image_0_t = np.transpose(image_0, (1, 2, 0))
plt.imshow(image_0_t[:,:,0], cmap='gray')

# sample 2
sample_2_np = sample_2.cpu().detach().numpy()
image_0 = sample_2_np[0]
image_0_t = np.transpose(image_0, (1, 2, 0))
plt.imshow(image_0_t[:,:,0], cmap='gray')

```

```bash
# check current error (mismatch in output and target sizes)
# loss = criterion(output, target)
/home/kartik/Pictures/Screenshots/Screenshot from 2023-07-09 01-37-28.png
```

## References

### MoBY

```
@article{xie2021moby,
  title={Self-Supervised Learning with Swin Transformers},
  author={Zhenda Xie and Yutong Lin and Zhuliang Yao and Zheng Zhang and Qi Dai and Yue Cao and Han Hu},
  journal={arXiv preprint arXiv:2105.04553},
  year={2021}
}
```

### Swin Transformer

```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
