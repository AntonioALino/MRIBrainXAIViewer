import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

CLASS_NAMES =  ["Glioma", "Meningioma", "Normal", "Pituitary Tumor"]

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class BrainMRIClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        # Backbone ResNet18 com pesos ImageNet como ponto de partida
            #refere-se a modelos de redes neurais pré-treinados, geralmente CNNs (Redes Neurais Convolucionais), usados como base para extrair características em tarefas complexas de visão computacional ou processamento de linguagem, permitindo reutilizar aprendizados prévios para novos objetivos de forma rápida e eficiente.
        self.backbone = models.resnet18(
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Todos os parâmetros treináveis (fine-tuning completo)
            #Técnica de aprendizado de máquina onde um modelo de Inteligência Artificial pré-treinado (como GPT, BERT ou CNNs) é retreinado em um conjunto de dados menor e específico.
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Substitui a cabeça de classificação original do ResNet18
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
        )

        # Buffers para Grad-CAM
            #Área de armazenamento temporário na memória (RAM) usada para guardar dados enquanto eles estão sendo transferidos de um lugar para outro, ou processados
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self) -> None:

        def _save_activations(module, input, output):
            self._activations = output.detach()

        def _save_gradients(module, grand_in, grad_out):
            self._gradients = grad_out.detach()

        self.backbone.layer4.register_forward_hook(_save_activations)
        self.backbone.layer4.register_full_backward_hook(_save_gradients)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

    @property
    def activations(self) -> torch.Tensor | None:
        return self._activations

    @property
    def gradients(self) -> torch.Tensor | None:
        return self._gradients


### --------------------- Loading Model ------------------###
def load_model(checkpoint_path: str) -> BrainMRIClassifier:
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de pesos não encontrado: {path.resolve()}\n"
            "Execute o train.py para gerar o brain_mri_weights.pth,\n"
            "ou consulte o README para outras opções."
        )

    model = BrainMRIClassifier(num_classes=len(CLASS_NAMES))

    state_dict = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state_dict)

    print(f"[model] Pesos carregados: {path.name}")
    print(f"[model] Classes: {CLASS_NAMES}")

    model.eval()
    return model

### --------------------- Pré Process ------------------###

def preprocess_image(img_rgb : np.ndarray) -> torch.Tensor:
    tensor = TRANSFORM(img_rgb)
    return tensor.unsqueeze(0)

