from n2n4m.crism_image import CRISMImage
from n2n4m.n2n4m_denoise import clip_bands
from n2n4m.preprocessing import impute_bad_values_in_image
import numpy as np

from main import RATIO_DATA
import onnxruntime

ort_session = onnxruntime.InferenceSession(
    "1024_test.onnx", providers=["CPUExecutionProvider"]
)


def get_pixel_data(file_path):
    # Im v sorry
    image = CRISMImage(file_path)
    image.ratio_image(RATIO_DATA)
    ratioed_im = image.ratioed_image
    ratioed_im = ratioed_im.reshape(-1, 438)  # 438 bands
    ratioed_im, _ = clip_bands(ratioed_im)
    ratioed_im = ratioed_im[:, :248]  # 248 bands to use for the model
    ratioed_im, _ = impute_bad_values_in_image(ratioed_im, threshold=10)
    min_vals = np.min(ratioed_im, axis=-1, keepdims=True)
    max_vals = np.max(ratioed_im, axis=-1, keepdims=True)
    ratioed_im_scaled = (ratioed_im - min_vals) / (
        (max_vals - min_vals) + 0.00001
    )

    return ratioed_im_scaled


def predict(data):
    input_dictionary = {"input.1": np.float32(data)}
    runtime_outputs = ort_session.run(None, input_dictionary)
    return runtime_outputs
