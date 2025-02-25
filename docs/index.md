# Welcome to CAML

The **C**RISM **A**nalysis with **M**achine **L**earning (CAML) software exists to mineral identification in CRISM imagery using machine learning. This system is designed to be easy to use (no-code), platform agnostic, and to provide a high level of accuracy in mineral identification. It is designed for use by planetary science researchers, students, and enthusiasts.

## Getting Started

To get started with CAML, you will need to install the system on your computer. This is available as a standalone desktop application, or as a Python package. The desktop application is recommended for users who are not familiar with Python, while the Python package is recommended for users who are familiar with Python and want to use CAML in their own scripts.

### Desktop Application

The desktop application is available for Windows, macOS, and Linux. To install the desktop application, download the appropriate installer from the [releases page](https://github.com/rob-platt/CRISM_classifier_application/releases) and follow the installation instructions.

### Python Package

The Python package is available on [Github](https://github.com/rob-platt/CRISM_classifier_application).

!!! warning
    The Python package is currently in development and is not yet available for general use.

!!! note
    This python package only releases code for running inference of the ML model. A separate, later release will include the model itself, preprocessing, and training code. 

## Citation

If you find this software useful, please cite the following paper:

```
@inproceedings{platt_automated_2024,
	title = {Automated Classification of CRISM Spectra Using Hybrid Neural Networks},
	language = {en},
	booktitle = {Tenth International Conference on Mars 2024},
    series = {LPI Contributions},
    volume = {3007},
	author = {Platt, R and Arcucci, R and John, C M},
	year = {2024},
    eid = {3166},
    pages = {3166},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2024LPICo3007.3166P},
}
```
