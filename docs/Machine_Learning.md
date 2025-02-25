# CAML Methodology

The classification model implemented in CAML is based on the model introduced in [Platt et al. 2024](https://www.hou.usra.edu/meetings/tenthmars2024/pdf/3166.pdf).

This is a hybrid neural network which combines a Variational Autoencoder (VAE) as a generative model, with a Convolutional Neural Network (CNN) to classify the spectra. The CNN uses the latent space from the VAE to predict the mineralogy per pixel. The output from the VAE (the "reconstructions") are the optional "reconstruction" spectra visualised in CAML. 

# Further Details
!!! note
    Further details on model training, performance, and validation - beyond those in [Platt et al. 2024](https://www.hou.usra.edu/meetings/tenthmars2024/pdf/3166.pdf) - will be provided in a future publication.