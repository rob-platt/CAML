import numpy as np
import onnxruntime
from n2n4m.n2n4m_denoise import clip_bands
from n2n4m.preprocessing import impute_bad_values_in_image

MODEL_PATH = "vae_classifier_1024.onnx"


class Classifier:
    """Classify the CRISM cube using the CRISM Classifier model."""

    def __init__(self):
        self.min_vals: np.ndarray = None
        self.max_vals: np.ndarray = None
        self.pred_cls: np.ndarray = None
        self.pred_conf: np.ndarray = None
        self.bad_pixels: np.ndarray[np.bool_] = None
        self.spatial_dims: tuple[int, int] = None
        self.recon: np.ndarray = None

    def preprocess_image(self, image) -> np.ndarray:
        """
        Preprocess the image for the model. Applies the following steps:
        - Clip image channels to 248 bands
        - Impute bad values (Ratioed I/F > 10) in the image
        - Scale the image between 0 and 1
        """
        image = image.reshape(-1, 438)  # 438 bands
        image, _ = clip_bands(image)
        image = image[:, :248]  # 248 bands to use for the model
        image, bad_pix = impute_bad_values_in_image(image, threshold=10)
        bad_pix = bad_pix.reshape(*self.spatial_dims, 248)
        self.bad_pixels = np.all(bad_pix, axis=-1)
        self.min_vals = np.min(image, axis=-1, keepdims=True)
        self.max_vals = np.max(image, axis=-1, keepdims=True)

        image_scaled = (image - self.min_vals) / (
            (self.max_vals - self.min_vals) + 0.00001
        )
        return image_scaled

    def inverse_normalise(self, image):
        """
        Inverse normalise the image to the original scale.
        """
        return image * (self.max_vals - self.min_vals) + self.min_vals

    def batch_array(self, array):
        """
        Split the image array into batches of 1024 pixels.
        If the image is not divisible by 1024, pad the last batch with zeros.

        Returns
        -------
        np.ndarray
            The image array split into batches
        int
            The number of pixels in the last batch that are "real" data
        """
        # Ensure the input array has the correct number of channels (248)
        if array.shape[1] != 248:
            raise ValueError("Input array must have 248 channels.")

        # Calculate the number of full blocks and the remainder
        num_full_blocks = array.shape[0] // 1024
        remainder = array.shape[0] % 1024

        # Split into full blocks
        blocks = [
            array[i * 1024 : (i + 1) * 1024] for i in range(num_full_blocks)
        ]

        # Handle the remainder by padding with zeros if necessary
        if remainder > 0:
            padded_block = np.zeros((1024, 248))
            padded_block[:remainder] = array[num_full_blocks * 1024 :]
            blocks.append(padded_block)

        return np.array(blocks, dtype="float32"), remainder

    def predict(self, image_array: np.ndarray):
        """
        Predict mineral classes and confidence scores for the input image.
        Also reconstruct the input image using the VAE model.
        """
        ort_session = onnxruntime.InferenceSession(
            (MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        image = np.empty_like(image_array)
        self.spatial_dims = image.shape[:-1]
        image[:] = image_array
        image = self.preprocess_image(image)
        batches, remainder = self.batch_array(image)
        pred_probs = []
        reconstructions = []
        for x in batches:
            onnx_input = x[:, np.newaxis, :]
            onnxruntime_input = {"input.1": onnx_input}
            reconstructions_batch, pred_probs_batch = ort_session.run(
                None, onnxruntime_input
            )
            reconstructions.append(reconstructions_batch)
            pred_probs.append(pred_probs_batch)

        reconstructions = np.array(reconstructions).reshape(-1, 248)
        pred_probs = np.array(pred_probs).reshape(-1, 38)
        # Remove the padding from the last batch if necessary
        if remainder > 0:
            pred_probs = pred_probs[: -(1024 - remainder)]
            reconstructions = reconstructions[: -(1024 - remainder)]

        reconstructions = self.inverse_normalise(reconstructions)
        reconstructions = reconstructions.reshape(*self.spatial_dims, 248)
        self.recon = reconstructions

        pred_probs = pred_probs.reshape(self.spatial_dims[:2] + (38,))
        self.pred_cls = np.argmax(pred_probs, axis=-1)
        self.pred_conf = np.max(pred_probs, axis=-1)

        # Set bad pixels to "artefact" class with -1 confidence
        self.pred_cls[self.bad_pixels] = 37
        self.pred_conf[self.bad_pixels] = -1
        return self.recon, self.pred_cls, self.pred_conf
