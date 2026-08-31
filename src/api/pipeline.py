import io
import numpy as np
import onnxruntime as otr
from PIL import Image


class DepthEstimator:
    def __init__(self, model_path: str):
        self.session = otr.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img_resized = image.convert("RGB").resize((384, 384))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        img_np = np.transpose(img_np, (2, 0, 1))
        return np.expand_dims(img_np, axis=0)

    def postprocess_to_png(self, depth_map: np.ndarray) -> bytes:
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-5:
            depth_norm = (depth_map - d_min) / (d_max - d_min) * 255.0
        else:
            depth_norm = np.zeros_like(depth_map)

        depth_uint8 = depth_norm.astype(np.uint8)
        depth_img = Image.fromarray(depth_uint8, mode="L")

        buf = io.BytesIO()
        depth_img.save(buf, format="PNG")
        return buf.getvalue()

    def predict(self, image_bytes: bytes) -> bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            input_tensor = self.preprocess(image)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        depth_map = outputs[0][0, 0]

        return self.postprocess_to_png(depth_map)
