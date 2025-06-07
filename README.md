## Overview
Source files to reproduces the main results of the PhD thesis entitled **Modelling of deuterium retention and desorption processes in tungsten under pulsed plasma and laser impact**.

### [![Latest PDF](https://img.shields.io/static/v1?label=PDF&logo=adobeacrobatreader&message=see%20latest%20version&color=success)](../../blob/build/dissertation.pdf)

## Structure
* [Thesis](Thesis): Thesis and annotation. I am very grateful to the authors of [Russian-Phd-LaTeX-Dissertation-Template](https://github.com/AndreyAkinshin/Russian-Phd-LaTeX-Dissertation-Template) that simplified significantly the process of my PhD preparation. To compile the files locally, visit the referenced Git repository for instruction on the work with the LaTeX template. 
* [Scripts](Scripts): FESTIM scripts and data to reproduce the results. Some raw data is too heavy to distribute via Git, so it was made available via Zenodo. 
    - [Chapter 1](./Scripts/Chapter_1): Scripts to reproduce several figures for the literature review
    - [Chapter 2](./Scripts/Chapter_2): Scripts and data for V&V of the kinetic surface model in FESTIM. The original scripts are openly available in this [repository](https://github.com/KulaginVladimir/FESTIM-SurfaceKinetics-Validation). More details can be found in the associated [paper](https://www.sciencedirect.com/science/article/abs/pii/S0360319925006937).
    - [Chapter 3](./Scripts/Chapter_3): Scripts and data for the model validation against QSPA-T experiments and simulations of the D retention in W under ELM-like exposure. The results on ELM-like impact on retention were published in [V.Kulagin et al. JNM 2025](https://www.sciencedirect.com/science/article/abs/pii/S0022311524004719).
    - [Chapter 4](./Scripts/Chapter_4/): Scripts and data for the model validation against [LID experiments](https://github.com/KulaginVladimir/LID-validation) and estimations of the LID efficiency under different conditions. The results on the estimation of the atomic fraction in the desorption flux are presented in [V.Kulagin et al. FusDes 2022](https://www.sciencedirect.com/science/article/pii/S0920379622002794) and [V.Kulagin et al. J. Surf. Investig.  2022](https://link.springer.com/article/10.1134/S1027451022050317). The results about the effect on material properties on the LID efficiency are published in [V.Kulagin et al. JNM 2023](https://www.sciencedirect.com/science/article/pii/S0022311523005147). 


## How to use

For a local use, clone the repository to your local machine.

```
git clone https://github.com/KulaginVladimir/PhD
```

Create and activate the correct conda environment with the required dependencies. Navigate to the [Scripts](./Scripts/) folder and run in your terminal:

```
conda env create -f environment.yml
conda activate PhD-env
```

This will set up a Conda environment named `PhD-env` with all the required dependencies for running the FESTIM scripts. 

> [!WARNING]  
> Some scripts for chapters 3 & 4 were ran on HPC with Slurm Workload Manager. Each subfolder includes the HPC scripts. An example on how to perform a sequential series of simulations is given in this [book](./Scripts/Chapter_4/LID_simulation/LID.ipynb)
> For any queries, contact: VVKulagin@mephi.ru

Navigate to the desired folder and run the Jupyter books using the activated Conda environment. The archive with the raw data can be accessed via Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14036908.svg)](https://doi.org/10.5281/zenodo.15614480)

> [!NOTE]  
> LaTeX is required to reproduce figures. To install required dependencies, run the following command in your terminal:
> ```
> sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
> ```

