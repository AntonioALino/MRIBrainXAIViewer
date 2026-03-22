import numpy as np
import cv2
import torch
import torch.nn.functional
from model.model import BrainMRIClassifier


class GradCAM:
    def __init__(self, model: BrainMRIClassifier) -> None:
        self.model = model


    def compute(
            self,
            input_tensor : torch.Tensor,
            target_class : int  | None = None,
            output_size: tuple [int, int]  = (224, 224),
    ) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        inp = input_tensor.clone().requires_grad_(True)
        logits = self.model(inp)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        gradients = self.model.gradients
        activations = self.model.activations

        if gradients is None or activations is None:
            raise RuntimeError(
                "Os hooks não capturaram tensores.\n"
                "Verifique se:\n"
                "  1. O modelo foi construído com BrainMRIClassifier()\n"
                "  2. O forward pass foi executado antes do compute()\n"
                "  3. O código não está dentro de torch.no_grad()"
            )

        weights = gradients.mean(dim=[2, 3]).squeeze()

        h, w = activations.shape[2], activations.shape[3]
        cam = torch.zeros(h, w, dtype=torch.float32)

        for k, alpha in enumerate(weights):
            cam += alpha * activations[0, k]

        cam = torch.nn.functional.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.detach().cpu().numpy()
        cam_resized = cv2.resize(
            cam_np,
            (output_size[1], output_size[0]),
            interpolation=cv2.INTER_CUBIC
        )

        cam_smooth = cv2.GaussianBlur(cam_resized, (11, 11), sigmaX=0)

        cam_smooth = np.clip(cam_smooth, 0.0, 1.0).astype(np.float32)

        return cam_smooth

### --------------------- Visualize Functions ------------------###

def apply_colormap(heatmap: np.ndarray) -> np.ndarray:

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

def blend_heatmap(
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
) -> np.ndarray:

    if original_rgb.shape[:2] != heatmap.shape[:2]:
        raise ValueError(
            f"Shapes incompatíveis: original={original_rgb.shape[:2]}, "
            f"heatmap={heatmap.shape[:2]}. "
            "Ambos devem ter a mesma altura e largura."
        )

    heatmap_rgb = apply_colormap(heatmap)

    blended = cv2.addWeighted(
        original_rgb, alpha,
        heatmap_rgb, 1.0 - alpha,
        0
    )

    return blended

def save_heatmap_png(
        heatmap: np.ndarray,
        output_path: str
) -> None:
    colored_rgb = apply_colormap(heatmap)
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, colored_bgr)




