import io
from PIL import Image
import numpy as np
import onnxruntime as otr
import matplotlib.pyplot as plt


class DepthEstimator:
    def __init__(self, model_path: str):
        self.session = otr.InferenceSession(
            model_path, provider_options=["CPUExecutionProvider"]
        )
        self.input_names = self.session.get_inputs()[0].name
        self.output_names = self.session.get_outputs()[0].name

    def preprocess(self, image: Image.Image):
        img_resized = image.convert("RGB").resize((384, 384))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)

        return img_np

    def predict(self, image_bytes: bytes):
        image = Image.open(io.BytesIO(image_bytes))

        input_tensor = self.preprocess(image)
        outputs = self.session.run(
            [self.output_names], {self.input_names: input_tensor}
        )
        depth_map = outputs[0][0, 0]

        plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.imshow(depth_map, cmap="magma")
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)

        return buf.getvalue()
