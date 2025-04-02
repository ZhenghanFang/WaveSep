# WaveSep
This repo contains the official PyTorch implementation for the paper [WaveSep: A Flexible Wavelet-based Approach for Source Separation in Susceptibility Imaging](https://link.springer.com/chapter/10.1007/978-3-031-44858-4_6), at [MLCN 2023](https://mlcnworkshop.github.io/)

by [Zhenghan Fang](https://zhenghanfang.github.io/), [Hyeong-Geol Shin](https://sites.google.com/view/hgshin-ptf/home?authuser=1), [Peter van Zijl](https://scholar.google.com/citations?user=mXPXyRgAAAAJ&hl=en), [Xu Li](https://scholar.google.com/citations?user=fakl0iYAAAAJ&hl=en), and [Jeremias Sulam](https://sites.google.com/view/jsulam)

## Dependencies
Create and activate a new conda environment
```
conda create -n wavesep python==3.10
conda activate wavesep
```
Install necessary python packages
```
pip install -r requirements.txt
```
<!-- Install the wavesep package
```
pip install -e .
``` -->


## Usage
### QSM source separation
```
python wavesep/qsm_sep.py --data <yml of input data>
``` 
The yml file contains the input data for QSM source separation. 
See data/yml/template_qsm.yml for more details.
See data/yml/example_qsm.yml for an example.

### STI source separation
```
python wavesep/sti_sep.py --data <yml of input data>
```
The yml file contains the input data for STI source separation. 
See data/yml/template_sti.yml for more details.
See data/yml/example_sti.yml for an example.

<!-- ## Citation
```
@article{WaveSep,
  title={WaveSep: A Flexible Wavelet-based Approach for Source Separation in Susceptibility Imaging},
  author={},
  journal={},
  year={2023}
}
``` -->

<!-- ## Acknowledgement
This work was supported by  -->

## Contact
If you have any questions, please contact me at
```
Zhenghan Fang

Email: zfang23@jhu.edu

```
