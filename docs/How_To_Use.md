# How To Use the Desktop Application

## Installation

Installation should be as simple as downloading the latest [release folder](https://github.com/rob-platt/CRISM_classifier_application/releases) for your operating system, unzipping the folder, and running the executable file.

## Selecting CRISM_ML Data

When you run CAMEL, the first thing you will be prompted for is the directory containing the CRISM_ML training dataset. This must be a folder, containing the `CRISM_bland_unratioed.mat` file for training the bland pixel classifier from [Plebani et al. 2022](https://doi.org/10.1016/j.icarus.2021.114849) ([Github Link](https://github.com/Banus/crism_ml)). This dataset can be found [here](https://zenodo.org/records/13338091).

!!! Warning
    CAMEL cannot load CRISM imagery without this dataset. The dataset must be named `CRISM_bland_unratioed.mat`

## Loading a CRISM Image

CAMEL is designed to work **solely with CRISM L sensor TRDR images with TRR3 processing**. Standard photometric and atmospheric corrections, including the "Volcano-Scan" atmospheric correction, are also expected to have been applied **before** loading the image into CAMEL. There are two methods for acquiring this data:  

- Downloading the data from the [PDS Geosciences Node](https://pds-geosciences.wustl.edu/missions/mro/crism.htm), and then applying the corrections using the [CRISM Analysis Toolkit](https://pds-geosciences.wustl.edu/missions/mro/crism.htm) (CAT) with ENVI software.
- Using [MarsSI](https://marssi.univ-lyon1.fr/MarsSI/map/mars/#0/0/0/MOLA) to source CRISM images with the desired corrections already applied. The data products from MarsSI with the `_CAT_corr.img` suffix are the ones to use. NB: The suffix must be removed to load the image into CAMEL.
```
Example: "FRT00009A16_07_IF166L_TRR3_CAT_corr.img" -> "FRT00009A16_07_IF166L_TRR3.img"
         "FRT00009A16_07_IF166L_TRR3_CAT_corr.hdr" -> "FRT00009A16_07_IF166L_TRR3.hdr"
         "FRT00009A16_07_IF166L_TRR3.lbl" -> "FRT00009A16_07_IF166L_TRR3.lbl"
``` 

The loading process may take up to a few minutes.

## Basic Visualisation Controls

After loading a CRISM image, you will be presented with the image alongside a plot of individual spectra. You can pan and zoom on the image plot using the controls underneath it. Hovering over a pixel 
will display the spectrum for that pixel in the plot. Left clicking on a pixel will keep that pixel displayed until you left click again.

The following controls are available:  

- Image Selection: Choose between visualising a ratioed image band, or a summary product (e.g. LCPINDEX2) created from the ratioed image.
- Image Channel Selection: If visualising an image band, select which band to display.
- Spectrum wavelength range: Choose the range of wavelengths to display in the spectrum plot.
- Classification: Button to run classification across the image using the CAMEL model. This will take a few minutes to complete.

!!! tip 
    For identification of most hydrated mineral features, plotting between 1.0 and 2.6 microns is recommended.

## Classification

After pressing the classification button, the CAMEL model will conduct pixel-wise classification across the image. This should be possible with almost any level of hardware[^1], as the model is relatively small. The classification process may take a few minutes to complete, depending on the size of the image and the hardware being used. 

## Advanced Visualisation Controls (Post-Classification)

After classification has been completed, the predicted mineralogy will be displayed on top of the image. The following additional controls will become available:

- Classification Results (On/Off): Toggle the mineralogy overlay on the image.
- Two filtering options:  
    * Connected Components: Minimum number of pixels of the same mineralogy in a contiguous region to display.
    * Minimum Confidence: Minimum confidence level to display prediction **averaged across the connected component**[^2].  
- Run Filtering: This will apply the chosen filtering options (together) to the classification results.

## Saving Results

### Saving Graphics

The image plot currently displayed can be saved as a .png file by using the :material-content-save: icon underneath the image plot. This will save the image with the current mineralogy overlay and any filtering applied.

### Saving CRISM Image

The CRISM image can be written out to an .img file using the `Save Image` button. This will save the original image to a new directory, with the mineral predictions as band 437, and the confidence levels as band 438. The current mineralogy overlay displayed are the predictions that will be saved, with all other pixels labelled as no data (65535 for CRISM data)[^3]. 

To allow for ENVI + CAT compatability (for map projection and further analysis), the image must be saved using the **exact** name of the original image file. Therefore please pick a new directory to save the image to, and ensure the original image file is not in that directory. The header (.hdr) and label (.lbl) files will also be saved. 



[^1]: Minimum 8GB RAM recommended. Currrently no GPU support.
[^2]: There are several ways these two filtering steps could be combined, we found this to give the most visually sensible results.
[^3]: This allows for easy visualisation in GIS software, without having to compute additional filtering. If you wish to have access to all of the predictions and filter in GIS, set the minimum confidence to 0.0 and the connected components to 0 before saving. 