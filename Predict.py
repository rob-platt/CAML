import numpy as np
import onnxruntime
from n2n4m.n2n4m_denoise import clip_bands
from n2n4m.preprocessing import impute_bad_values_in_image

MODEL_PATH = "vae_classifier_1024.onnx"


class Classifier:
    """Classify the CRISM cube using the CRISM Classifier model."""

    def __init__(self):
        self.pred_cls: np.ndarray = None
        self.pred_conf: np.ndarray = None

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
        image, _ = impute_bad_values_in_image(image, threshold=10)
        min_vals = np.min(image, axis=-1, keepdims=True)
        max_vals = np.max(image, axis=-1, keepdims=True)

        image_scaled = (image - min_vals) / ((max_vals - min_vals) + 0.00001)
        return image_scaled

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
        """
        ort_session = onnxruntime.InferenceSession(
            (MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        image = np.empty_like(image_array)
        original_shape = image.shape
        image[:] = image_array
        image = self.preprocess_image(image)
        batches, remainder = self.batch_array(image)
        pred_probs = []
        for x in batches:
            onnx_input = x[:, np.newaxis, :]
            onnxruntime_input = {"input.1": onnx_input}
            pred_probs.append(ort_session.run(None, onnxruntime_input)[1])

        pred_probs = np.array(pred_probs).reshape(-1, 38)
        # Remove the padding from the last batch if necessary
        if remainder > 0:
            pred_probs = pred_probs[: -(1024 - remainder)]
        pred_probs = pred_probs.reshape(original_shape[:2] + (38,))
        self.pred_cls = np.argmax(pred_probs, axis=-1)
        self.pred_conf = np.max(pred_probs, axis=-1)
        return self.pred_cls, self.pred_conf
