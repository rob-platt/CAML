# Welcome to CAML

The **C**RISM **A**nalysis with **M**achine **L**earning (CAML) software exists to mineral identification in CRISM imagery using machine learning. This system is designed to be easy to use (no-code), platform agnostic, and to provide a high level of accuracy in mineral identification. It is designed for use by planetary science researchers, students, and enthusiasts.

## Getting Started

To get started with CAML, you will need to install the system on your computer. This is available as a standalone desktop application, or as a Python package. The desktop application is recommended for users who are not familiar with Python, while the Python package is recommended for users who are familiar with Python and want to use CAML in their own scripts.

### Desktop Application

The desktop application is available for Windows, macOS, and Linux. To install the desktop application, download the appropriate installer from the [releases page](https://github.com/rob-platt/CRISM_classifier_application/releases) and follow the installation instructions for your operating system.

#### Windows

Download the "WindowsOS_CAML_vX_X_X.zip" file from the [releases page](https://github.com/rob-platt/CRISM_classifier_application/releases). Extract the contents of the zip file to a folder on your computer. Run the "CAML.exe" file to start the application. Alternatively, you can start the program from the command line by navigating to the folder where you extracted the files and running the following command:

```bash
& .\CAML.exe
```

#### macOS

Download the "macOS_CAML_vX_X_X.zip" file from the [releases page](https://github.com/rob-platt/CRISM_classifier_application/releases). Extract the contents of the zip file to a folder on your computer. From the command line, navigate to the folder where you extracted the files and running the following commands:

```bash
xattr -r -d com.apple.quarantine _internal/*
./CAML
```

#### Linux

Download the "LinuxOS_CAML_vX_X_X.zip" file from the [releases page](https://github.com/rob-platt/CRISM_classifier_application/releases). Extract the contents of the zip file to a folder on your computer. Start the program from the command line by navigating to the folder where you extracted the files and running the following command:

```bash
./CAML
```

### Python Package

The Python package is available on [Github](https://github.com/rob-platt/CRISM_classifier_application).

To create a new conda environment and install the package, run the following commands:

```bash
conda env create -f environment.yml
conda activate caml
```

Or you can install the dependencies into a virtual environment of your choice using the `requirements.txt` file:

```bash
# Activate your virtual environment
pip install -r requirements.txt
```
To run the application, use the following command:

```bash
python main.py
```

!!! note
    This python package only releases code for running inference of the ML model. A separate, later release will include the model itself, preprocessing, and training code. 

## Citation

If you find this software useful, please cite the following paper:

Platt, R., Arcucci, R. & John, C.M., 2024. Automated classification of CRISM spectra using hybrid neural networks. *Tenth International Conference on Mars 2024*, LPI Contributions, 3007, p.3166. Available at: https://ui.adsabs.harvard.edu/abs/2024LPICo3007.3166P.

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
