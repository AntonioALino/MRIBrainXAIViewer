# 🧠 MRI Brain XAI Viewer

> Sistema interativo para análise de imagens de ressonância magnética (MRI) cerebral com **Deep Learning**, **Explainable AI (Grad-CAM)** e **visualização 3D** no browser, com controle por gestos via webcam.

---

## ✨ Funcionalidades

| Funcionalidade | Status |
|---|---|
| Upload de imagem MRI | ✅ |
| Classificação com ResNet18 fine-tuned (~95% acurácia) | ✅ |
| Heatmap Grad-CAM (Explainable AI) | ✅ |
| Visualização 3D com Three.js | ✅ |
| Rotação e zoom via mouse | ✅ |
| Clique em regiões cerebrais com info clínica | ✅ |
| Comparação lado a lado de dois pacientes | ✅ |
| Controle gestual via webcam (MediaPipe) | ✅ |

**Classes classificadas:**
- 🟢 Normal
- 🔴 Glioma
- 🟡 Meningioma
- 🟠 Pituitary Tumor (Tumor Hipofisário)

---

## 🏗️ Arquitetura

```
mri-brain-system/
├── backend/                   # Python · FastAPI
│   ├── app.py                 # Endpoints REST: /analyze, /compare, /brain-region
│   ├── model.py               # Classificador ResNet18 com PyTorch
│   ├── gradcam.py             # Implementação Grad-CAM (XAI)
│   ├── brain_regions.py       # Atlas cerebral + mapeamento UV
│   ├── brain_mri_weights.pth  # Pesos treinados (ver seção abaixo)
│   └── requirements.txt
│
└── frontend/                  # TypeScript · Vite
    ├── index.html
    ├── vite.config.ts
    └── src/
        ├── main.ts            # Orquestrador principal
        ├── scene.ts           # Three.js: cena 3D, texturas, raycasting
        ├── handTracking.ts    # MediaPipe: detecção de gestos
        ├── api.ts             # Cliente HTTP tipado
        ├── brainRegions.ts    # Painel de informações anatômicas
        └── style.css
```

### Fluxo de dados

```
[Upload MRI] → POST /analyze → [ResNet18 fine-tuned — forward pass]
                             → [Grad-CAM] → heatmap PNG (base64)
                             → [Classificação] → label + confiança
                             ↓
[Three.js] → textura MRI + overlay Grad-CAM no PlaneGeometry
           → OrbitControls (mouse) / HandTracker (webcam)
           → Raycasting → UV coords → POST /brain-region/coords
           → Painel anatômico com função e relevância clínica
```

---

## 🛠️ Stack

**Backend**
- Python 3.10+
- FastAPI + Uvicorn
- PyTorch + TorchVision (ResNet18 fine-tuned)
- OpenCV
- Grad-CAM (implementação própria)

**Frontend**
- TypeScript + Vite
- Three.js (visualização 3D)
- MediaPipe Tasks Vision (hand tracking)

---

## 🚀 Instalação e execução

### Pré-requisitos

- Python 3.10 ou superior
- Node.js 18 ou superior
- Git
- Arquivo de pesos `brain_mri_weights.pth` (ver seção **Modelo**)

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/mri-brain-system.git
cd mri-brain-system
```

### 2. Adicione os pesos do modelo

Coloque o arquivo `brain_mri_weights.pth` dentro da pasta `backend/` antes de iniciar o servidor. Veja a seção **Modelo** para saber como obtê-lo.

### 3. Backend

```bash
cd backend
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

O servidor estará disponível em `http://localhost:8000`.
Documentação interativa (Swagger): `http://localhost:8000/docs`

> **Sem GPU local?** Instale a versão CPU do PyTorch:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```
> O modelo roda perfeitamente em CPU para inferência — apenas o treinamento requer GPU.

### 4. Frontend

```bash
# Em um novo terminal, a partir da raiz do projeto
cd frontend
npm install
npm run dev
```

A aplicação estará disponível em `http://localhost:3000`.

---

## 🤖 Modelo de Deep Learning

O sistema usa um **ResNet18 com fine-tuning completo**, treinado no dataset público de tumores cerebrais do Kaggle.

### Dataset

[Brain Tumor Classification MRI — Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

- ~3.200 imagens de MRI cerebral
- 4 classes: Glioma, Meningioma, Normal, Pituitary Tumor
- Dividido em Training e Testing

### Arquitetura

```
ResNet18 (backbone pré-treinado ImageNet)
  └── layer4 ← hooks de Grad-CAM registrados aqui
       └── FC head fine-tuned:
             Linear(512 → 256) → ReLU → Dropout(0.5) → Linear(256 → 4)
```

### Desempenho

| Métrica | Valor |
|---|---|
| Acurácia de validação | ~95–97% |
| Epochs de treinamento | 25 |
| Otimizador | Adam (lr=1e-4, weight decay=1e-5) |
| Scheduler | CosineAnnealingLR |
| Data augmentation | Flip, rotação ±20°, ColorJitter |

### Obtendo os pesos treinados

Os pesos não estão incluídos no repositório por causa do tamanho (~45 MB). Você tem duas opções:

**Opção A — Treinar você mesmo (Google Colab gratuito, ~2h)**

1. Acesse [colab.research.google.com](https://colab.research.google.com) e ative a GPU T4 em **Runtime → Change runtime type → T4 GPU**
2. Baixe o dataset do Kaggle e faça upload para o Google Drive
3. Execute o `train.py` disponível na pasta `backend/`
4. Baixe o arquivo `brain_mri_weights.pth` gerado
5. Coloque-o dentro da pasta `backend/`

**Opção B — Entrar em contato**

Abra uma [issue](https://github.com/AntonioALino/MRIBrainXAIViewer/issues) solicitando o arquivo de pesos pré-treinados.

### Ordem das classes

O modelo segue a ordenação **alfabética** das pastas do dataset (padrão `ImageFolder` do PyTorch):

```
0 → Glioma
1 → Meningioma
2 → Normal (notumor)
3 → Pituitary Tumor
```

---

## 🖐️ Hand Tracking

O controle gestual usa **MediaPipe Hands** diretamente no browser — sem dependência de Python para visão computacional, processamento 100% local na GPU do cliente.

**Como usar:**
1. Clique em **"✋ Ativar Gestos"** na sidebar
2. Permita o acesso à câmera quando solicitado
3. **Mão aberta + mover** → rotaciona o plano 3D
4. **Punho fechado** → pausa a rotação

O preview da câmera com o skeleton da mão aparece no canto inferior direito.

---

## 🌐 API REST

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/analyze` | Analisa uma MRI: classificação + Grad-CAM |
| `POST` | `/compare` | Compara dois exames lado a lado |
| `POST` | `/brain-region/coords` | Região cerebral por coordenadas UV |
| `GET` | `/brain-region/{id}` | Info de região por ID |
| `GET` | `/health` | Health check |

**Exemplo — `/analyze`:**

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@minha_mri.jpg"
```

Resposta:
```json
{
  "original": "<base64 PNG>",
  "heatmap":  "<base64 PNG>",
  "overlay":  "<base64 PNG>",
  "classification": {
    "label": "Glioma",
    "predicted_class": 0,
    "confidence": 0.9312,
    "probabilities": {
      "Glioma": 0.9312,
      "Meningioma": 0.0421,
      "Normal": 0.0198,
      "Pituitary Tumor": 0.0069
    }
  }
}
```

---

## 🧬 Regiões Cerebrais Mapeadas

| Região | Clique em... | Informações |
|---|---|---|
| Lobo Frontal | Terço superior da imagem | Planejamento, controle motor, fala (Broca) |
| Lobo Parietal | Centro da imagem | Integração sensorial, orientação espacial |
| Lobo Temporal Esq. | Borda esquerda, meio | Compreensão da linguagem (Wernicke) |
| Lobo Temporal Dir. | Borda direita, meio | Reconhecimento de faces, prosódia |
| Lobo Occipital | Terço inferior | Processamento visual |
| Corpo Caloso | Centro exato | Conexão entre hemisférios |
| Tronco Cerebral | Centro-baixo | Funções vitais, nervos cranianos |
| Cerebelo | Terço inferior | Coordenação motora, equilíbrio |

---

## 🔬 Explainability — Grad-CAM

O sistema usa **Gradient-weighted Class Activation Mapping** para explicar as predições da rede neural:

1. **Forward pass** → captura ativações da `layer4` do ResNet18 via hook
2. **Backward** do score da classe predita → captura gradientes via hook
3. **Global Average Pooling** dos gradientes → pesos de importância por canal
4. **Combinação linear ponderada** das ativações → mapa de calor bruto
5. **ReLU + resize + suavização gaussiana** → heatmap final sobreposto à MRI

O heatmap indica *onde* o modelo focou para tomar sua decisão. Regiões quentes (vermelho/amarelo) foram determinantes para a classificação — permitindo auditabilidade da predição.

---

## 📊 Métricas de avaliação

**Classificação:**
- Acurácia, Precisão, Recall, F1-Score (macro e weighted)
- AUC-ROC por classe (One-vs-Rest)
- Matriz de confusão normalizada
- Cohen's Kappa (concordância com especialistas)

**Explainability (XAI):**
- Pointing Game Accuracy (heatmap vs anotação ground-truth)
- IoU (máscara do heatmap vs segmentação)
- Insertion/Deletion AUC (fidelidade da explicação)

**Interface (HCI):**
- System Usability Scale (SUS)
- NASA-TLX (carga cognitiva)
- Task Completion Time

---

## 🗺️ Roadmap

- [ ] Suporte a volumes NIfTI (cortes axial/sagital/coronal interativos)
- [ ] Segmentação de tumores com U-Net
- [ ] Atlas cerebral com registro MNI (coordenadas reais por voxel)
- [ ] GradCAM++ e SHAP para comparação de métodos XAI
- [ ] Exportação de relatório PDF por paciente
- [ ] Tracking longitudinal (série temporal de exames)
- [ ] Suporte multimodal (T1, T2, FLAIR, T1-Gd)

---

## ⚠️ Aviso

Este sistema é um **protótipo acadêmico** desenvolvido para fins de pesquisa e demonstração. **Não deve ser utilizado para diagnóstico clínico real.** Para uso médico, consulte sempre um profissional de saúde qualificado.

---

## 📄 Licença

MIT License — veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

Desenvolvido por Antônio Lino

- LinkedIn: [Antônio Lino]([(https://www.linkedin.com/in/antonio-augusto-prado-lino)])

---

*Desenvolvido como projeto de pesquisa em Inteligência Artificial aplicada à neuroimagem.*
