# FS_MedSAM2: Exploring the Potential of SAM2 for Few-Shot Medical Image Segmentation without Fine-tuning

[arxiv](https://arxiv.org/abs/2409.04298)

## News

We provide `example_data` folder, which contains some example data for anyone who want to try this code.


## Getting Strated

### Data preparation

You can obtain the data by following steps, or directly obtain the data by [Baidu Cloud](https://pan.baidu.com/s/1DPqK3Qps0h5ZU2EzD1kSwg?pwd=t9vu) (password: t9vu).

1. Follow [SSL_ALPNet](https://github.com/zmcheng9/GMRD) to create SSL_ALPNet project at  `/path/to/SSL_ALPNet`;
2. Process Synapse-CT and CHAOS-MRI datasets follow SSL_ALPNet;
3. Place `validation_wopred.py` in the `/path/to/SSL_ALPNet` directory, update the `saved_npz_path = '/path/to/saved_npz'`, and run it using either `/path/to/SSL_ALPNet/test_ssl_abdominal_ct.sh` or `/path/to/SSL_ALPNet/test_ssl_abdominal_mri.sh`.

### Experiments Reproduction

1. Based on the official repository of [SAM2](https://github.com/facebookresearch/segment-anything-2), deploy SAM2 locally at `/path/to/SAM2`;
2. Move the files and folders from this repository, `/sam2` and those under `/notebooks`, to the corresponding folder at `/path/to/SAM2/sam2` and `/path/to/SAM2/notebooks`. e.g, move `/FS_MedSAM2/sam2/build_fsmedsam2.py` under `/path/to/SAM2/sam2/build_fsmedsam2.py`.
3. After changing the `saved_npz_path = '/path/to/saved_npz'` and `ckpt_path = '/path/to/ckpt'`, eval FS_MedSAM2:
```
cd /path/to/SAM2/notebooks
python infer_fsmedsam2_by_slice.py # infer 1S1Q
python infer_fsmedsam2_by_volume.py # infer S1SFQ from top
python infer_fsmedsam2_by_volume_from_middle.py # infer S1SFQ from middle
```

## Citation

If you find these projects useful, please consider citing:

```bibtex
@misc{bai2024fsmedsam2exploringpotentialsam2,
      title={FS-MedSAM2: Exploring the Potential of SAM2 for Few-Shot Medical Image Segmentation without Fine-tuning}, 
      author={Yunhao Bai and Qinji Yu and Boxiang Yun and Dakai Jin and Yingda Xia and Yan Wang},
      year={2024},
      eprint={2409.04298},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.04298}, 
}

```


## Questions
If you have any questions, welcome contact me at 'yhbai@stu.ecnu.edu.cn'
