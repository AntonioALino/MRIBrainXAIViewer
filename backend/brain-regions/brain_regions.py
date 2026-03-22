from __future__ import annotations

from pandas.conftest import ordered

RegionDict = dict[str, object]

BRAIN_REGIONS: dict[str, RegionDict] = {

# ── Lobo Frontal ──────────────────────────────────────────────────────────
    "frontal_lobe": {
        "name": "Lobo Frontal",

        "function": (
            "O lobo frontal é o maior lobo cerebral e o centro executivo do cérebro. "
            "O córtex pré-frontal dorsolateral coordena planejamento, tomada de decisões, "
            "raciocínio abstrato e memória de trabalho. O córtex motor primário (giro pré-central) "
            "controla movimentos voluntários do hemicorpo contralateral de forma somatotópica — "
            "a representação homuncular de Penfield mapeia cada região corporal a uma área "
            "específica do córtex. A área de Broca (giro frontal inferior esquerdo, áreas 44 e 45 "
            "de Brodmann) é essencial para a produção articulada da fala e a sintaxe linguística. "
            "O córtex orbitofrontal regula o controle inibitório de impulsos e o julgamento social."
        ),

        "clinical_relevance": (
            "Lesões no córtex motor primário causam hemiparesia ou hemiplegia contralateral "
            "com sinal de Babinski positivo (lesão do neurônio motor superior). "
            "Lesões na área de Broca causam afasia de Broca: fala não-fluente, telegráfica, "
            "com compreensão relativamente preservada — o paciente sabe o que quer dizer "
            "mas não consegue articular. "
            "Lesões pré-frontais produzem síndrome disexecutiva: dificuldade de planejamento, "
            "impulsividade, perseveração (repetição involuntária de respostas) e alterações "
            "de personalidade como as descritas no caso clássico de Phineas Gage (1848). "
            "Tumores frontais frequentemente crescem silenciosamente até atingir grande volume "
            "antes de causar sintomas, devido à plasticidade funcional das regiões adjacentes."
        ),

        "pathologies": [
            "Glioblastoma (GBM) — tumor primário mais comum em adultos",
            "Meningioma da convexidade frontal",
            "AVC da artéria cerebral anterior (ACA)",
            "Metástases (mama, pulmão, melanoma)",
            "Contusão frontal por trauma cranioencefálico",
        ],

        "brodmann_areas": ["4 (motor primário)", "6 (pré-motor)", "8", "9", "10",
                           "11 (orbitofrontal)", "44 (Broca)", "45 (Broca)"],

        "vascularization": "Artéria cerebral média (MCA) — território lateral; "
                           "Artéria cerebral anterior (ACA) — face medial",

        "color": "#4A90D9",

        # Coordenadas UV no espaço normalizado [0,1] do corte axial
        # u_min/u_max = faixa horizontal | v_min/v_max = faixa vertical
        "coords": {
            "u_min": 0.15, "u_max": 0.85,
            "v_min": 0.00, "v_max": 0.32,
        },
    },

    # ── Lobo Parietal ─────────────────────────────────────────────────────────
    "parietal_lobe": {
        "name": "Lobo Parietal",

        "function": (
            "O lobo parietal integra informações sensoriais somáticas — tato, pressão, "
            "temperatura, dor e propriocepção — provenientes do hemicorpo contralateral. "
            "O córtex somatossensorial primário (giro pós-central, áreas 1, 2 e 3 de Brodmann) "
            "possui organização somatotópica análoga ao córtex motor. "
            "O lobo parietal posterior (áreas 5 e 7) realiza integração multimodal: combina "
            "informações visuais, somáticas e vestibulares para construir a representação "
            "espacial do corpo e do ambiente — essencial para coordenação visuomotora, "
            "atenção espacial e manipulação de objetos. "
            "O giro angular (área 39) e o giro supramarginal (área 40) do hemisfério dominante "
            "participam da leitura, escrita, cálculo e nomeação."
        ),

        "clinical_relevance": (
            "Lesões do lobo parietal dominante (esquerdo em diestros) causam a síndrome de "
            "Gerstmann: acalculia (incapacidade de calcular), agrafia (sem alexia), "
            "agnosia digital (não reconhece os próprios dedos) e desorientação "
            "direita-esquerda — tetrada patognomônica de lesão do giro angular esquerdo. "
            "Lesões do lobo parietal não-dominante causam síndrome de negligência "
            "contralateral (hemineglect): o paciente ignora completamente o hemiespaço "
            "esquerdo — não come metade do prato, não barbeia metade do rosto. "
            "Anosognosia (negação da própria hemiplegia) e apraxia construcional "
            "completam o quadro da lesão parietal direita."
        ),

        "pathologies": [
            "Glioma parietal de alto grau",
            "Metástases — localização frequente",
            "AVC da artéria cerebral média (ramos parietais)",
            "Meningioma parasagital",
        ],

        "brodmann_areas": ["1, 2, 3 (somatossensorial)", "5", "7",
                           "39 (giro angular)", "40 (giro supramarginal)"],

        "vascularization": "Artéria cerebral média (MCA) — ramos parietais superiores e inferiores",

        "color": "#7B68EE",

        "coords": {
            "u_min": 0.15, "u_max": 0.85,
            "v_min": 0.30, "v_max": 0.52,
        },
    },

    # ── Lobo Temporal Esquerdo ────────────────────────────────────────────────
    "temporal_lobe_left": {
        "name": "Lobo Temporal Esquerdo",

        "function": (
            "O lobo temporal esquerdo (dominante em ~96% dos destros) abriga as estruturas "
            "neurais mais críticas para a linguagem receptiva. "
            "A área de Wernicke (giro temporal superior posterior, área 22 de Brodmann) "
            "processa e decodifica a linguagem auditiva: transforma sequências fonológicas "
            "em representações semânticas. "
            "O fascículo arqueado conecta Wernicke (temporal) a Broca (frontal), "
            "formando o circuito perisilviano da linguagem. "
            "O hipocampo e o córtex entorrinal, nas profundidades do lobo temporal, "
            "são a interface entre memória de curto e longo prazo — essenciais para a "
            "consolidação de novas memórias episódicas e semânticas. "
            "O córtex temporal inferior processa formas visuais complexas e identidade de objetos."
        ),

        "clinical_relevance": (
            "Lesões da área de Wernicke causam afasia de Wernicke: fala fluente, "
            "melodiosa, mas vazia de conteúdo semântico — parafasias (trocas de palavras), "
            "neologismos, jargonofasia. O paciente não compreende o que lhe dizem nem "
            "percebe seus próprios erros (anosognosia para a afasia). "
            "Lesões hipocampais bilaterais causam amnésia anterógrada severa — "
            "a síndrome de HM (Henry Molaison) é o caso mais estudado na neurociência. "
            "Epilepsia do lobo temporal (ELT) é a epilepsia focal mais comum no adulto: "
            "auras olfativas/gustativas, automatismos oroalimentares (mastigação, deglutição), "
            "déjà vu. A esclerose hipocampal é a causa mais frequente de ELT refratária."
        ),

        "pathologies": [
            "Esclerose hipocampal mesial — causa mais comum de ELT",
            "DNET (tumor neuroepitelial disembrioplásico) — benigno, epileptogênico",
            "Ganglioglioma — tumor misto, frequente em jovens com epilepsia",
            "Glioma de baixo grau",
            "AVC da artéria cerebral posterior (território hipocampal)",
        ],

        "brodmann_areas": ["21 (temporal médio)", "22 (Wernicke)", "28 (entorrinal)",
                           "37", "38 (polo temporal)"],

        "vascularization": "Artéria cerebral média (MCA) — temporal superior; "
                           "Artéria cerebral posterior (PCA) — hipocampo e giro parahipocampal",

        "color": "#E67E22",

        # Borda esquerda da imagem — convenção radiológica:
        # esquerda da imagem = hemisfério direito do paciente (e vice-versa)
        # Aqui mapeamos "esquerda da tela" como temporal esquerdo do PACIENTE
        # conforme convenção neurológica (não radiológica)
        "coords": {
            "u_min": 0.00, "u_max": 0.22,
            "v_min": 0.28, "v_max": 0.68,
        },
    },

    # ── Lobo Temporal Direito ─────────────────────────────────────────────────
    "temporal_lobe_right": {
        "name": "Lobo Temporal Direito",

        "function": (
            "O lobo temporal direito (não-dominante) é especializado em processamento "
            "auditivo não-verbal e reconhecimento de padrões complexos. "
            "Processa a prosódia da fala — entonação, ritmo e carga emocional das palavras — "
            "e é o substrato neural para percepção musical (amusia quando lesado). "
            "O giro fusiforme direito (área fusiforme de faces, FFA) é especializado no "
            "reconhecimento de faces familiares — uma das capacidades cognitivas mais robustas "
            "e evolutivamente antigas do ser humano. "
            "A amígdala direita coordena respostas emocionais a estímulos visuais e auditivos, "
            "especialmente ao medo e à ameaça. "
            "A memória visuoespacial e topográfica (lembrar lugares, rotas, mapas) "
            "depende predominantemente do hipocampo direito."
        ),

        "clinical_relevance": (
            "Lesões do giro fusiforme direito causam prosopagnosia: incapacidade de "
            "reconhecer rostos familiares, incluindo o próprio rosto no espelho — "
            "o paciente reconhece as pessoas pela voz, roupas ou cabelo. "
            "Lesões da amígdala direita embotam o reconhecimento de expressões de medo "
            "e raiva, com implicações para comportamento social e empatia. "
            "Lesões do hipocampo direito causam déficits de memória topográfica: "
            "incapacidade de aprender novos trajetos ou de se localizar em ambientes "
            "novos (síndrome de desorientação topográfica). "
            "Epilepsia temporal direita manifesta-se com auras visuais complexas, "
            "ilusões perceptuais e experiências de despersonalização/desrealização."
        ),

        "pathologies": [
            "Glioma temporal direito",
            "Meningioma da asa do esfenoide",
            "Cavernoma temporal",
            "AVC da artéria cerebral média",
            "Metástases",
        ],

        "brodmann_areas": ["21 (temporal médio)", "22 (temporal superior)",
                           "37 (fusiforme)", "38 (polo temporal)"],

        "vascularization": "Artéria cerebral média (MCA) — temporal superior; "
                           "Artéria cerebral posterior (PCA) — hipocampo",

        "color": "#D35400",

        "coords": {
            "u_min": 0.78, "u_max": 1.00,
            "v_min": 0.28, "v_max": 0.68,
        },
    },

    # ── Lobo Occipital ────────────────────────────────────────────────────────
    "occipital_lobe": {
        "name": "Lobo Occipital",

        "function": (
            "O lobo occipital é o centro primário do processamento visual. "
            "O córtex visual primário (V1, área 17 de Brodmann, estriado) localiza-se "
            "nas margens do sulco calcarino e recebe projeções diretas da retina "
            "via corpo geniculado lateral do tálamo — uma via de apenas dois neurônios "
            "entre a retina e o córtex. V1 é retinotópico: cada posição no campo visual "
            "corresponde a um ponto específico em V1. "
            "As áreas visuais de associação formam duas vias funcionais: "
            "via ventral (V4 → córtex temporal inferior) processa cor, forma e identidade de "
            "objetos — o 'o que é?'; "
            "via dorsal (V5/MT → córtex parietal posterior) processa movimento e "
            "localização espacial — o 'onde está e como pegar?'."
        ),

        "clinical_relevance": (
            "Lesões unilaterais de V1 causam hemianopsia homônima contralateral: "
            "perda da metade do campo visual de ambos os olhos no lado oposto à lesão. "
            "Lesões bilaterais causam cegueira cortical (síndrome de Anton em casos graves: "
            "o paciente é cego mas nega a cegueira). "
            "Lesões de V4 causam acromatopsia cerebral: visão em escala de cinza. "
            "Lesões de V5/MT causam acinetopsia: o paciente vê o mundo como uma série "
            "de fotografias estáticas — não percebe movimento contínuo. "
            "Isquemia da artéria cerebral posterior (ACP) é a causa mais comum de lesão "
            "occipital; enxaqueca com aura visual é a causa benigna mais frequente."
        ),

        "pathologies": [
            "AVC da artéria cerebral posterior (ACP)",
            "Meningioma occipital da foice",
            "Glioma occipital",
            "Displasia cortical focal",
        ],

        "brodmann_areas": ["17 (V1 — estriado)", "18 (V2 — parastriado)",
                           "19 (V3/V4/V5 — peristriado)"],

        "vascularization": "Artéria cerebral posterior (PCA) — ramo da artéria basilar",

        "color": "#27AE60",

        "coords": {
            "u_min": 0.20, "u_max": 0.80,
            "v_min": 0.65, "v_max": 0.85,
        },
    },

    # ── Corpo Caloso ──────────────────────────────────────────────────────────
    "corpus_callosum": {
        "name": "Corpo Caloso",

        "function": (
            "O corpo caloso é a maior comissura cerebral: um feixe compacto de "
            "aproximadamente 200 milhões de axônios mielinizados que conecta regiões "
            "homólogas e heterólogas dos dois hemisférios cerebrais. "
            "Anatomicamente divide-se em quatro regiões com conectividades distintas: "
            "rostro (conecta córtices pré-frontais orbitais), joelho ou genu "
            "(conecta córtices pré-frontais dorsolaterais), corpo (conecta córtices "
            "motor, somatossensorial e parietal) e esplênio (conecta córtices "
            "temporal posterior, parietal inferior e occipital). "
            "Funcionalmente, coordena a transferência bidirecional de informação "
            "sensorial, motora e cognitiva entre os hemisférios, permitindo integração "
            "das especializações hemisféricas e coordenação motora bilateral fina."
        ),

        "clinical_relevance": (
            "Lesões do corpo caloso produzem a síndrome de desconexão inter-hemisférica: "
            "mão alienígena (a mão esquerda realiza ações autônomas não-intencionadas), "
            "conflito inter-manual, alexia sem agrafia (lesão do esplênio: lê com o "
            "hemicampo direito mas a informação não chega ao hemisfério esquerdo para "
            "interpretação linguística). "
            "O padrão em 'borboleta' ou 'asa de borboleta' no corte axial — tumor que "
            "cruza a linha média pelo corpo caloso — é altamente sugestivo de "
            "glioblastoma (GBM) ou linfoma primário do SNC, ambos com pior prognóstico "
            "cirúrgico por impossibilidade de ressecção completa. "
            "Desmielinização do corpo caloso é marcador de esclerose múltipla: "
            "as lesões calosas perpendiculares ao eixo longo ('dedos de Dawson') "
            "são patognomônicas na RM sagital em FLAIR."
        ),

        "pathologies": [
            "Glioblastoma em 'borboleta' — cruza a linha média",
            "Linfoma primário do SNC — frequentemente periventricular e caloso",
            "Esclerose múltipla — 'dedos de Dawson' em FLAIR sagital",
            "Encefalopatia de Marchiafava-Bignami — desmielinização calosa por etilismo",
            "Agênese do corpo caloso — malformação congênita",
        ],

        "brodmann_areas": ["N/A — substância branca comissural (não cortical)"],

        "vascularization": "Artéria pericallosa (ramo da ACA) — superfície superior; "
                           "Artéria cerebral posterior (PCA) — esplênio",

        "color": "#C0392B",

        # Estrutura mediana estreita — coordenadas centrais com faixa estreita
        "coords": {
            "u_min": 0.35, "u_max": 0.65,
            "v_min": 0.38, "v_max": 0.58,
        },
    },

    # ── Tronco Cerebral ───────────────────────────────────────────────────────
    "brainstem": {
        "name": "Tronco Cerebral",

        "function": (
            "O tronco cerebral é a estrutura mais crítica do sistema nervoso central: "
            "uma coluna de apenas ~7,5 cm que concentra a maioria das funções vitais. "
            "Divide-se em três segmentos com funções distintas: "
            "Mesencéfalo (porção superior): controla movimentos oculares (nervos III e IV), "
            "pupila (núcleo de Edinger-Westphal) e tônus muscular (via reticuloespinhal). "
            "Abriga a substância negra (dopaminérgica — degenerada no Parkinson) e "
            "o núcleo rubro (coordenação motora). "
            "Ponte (pons): maior porção; contém núcleos dos nervos V, VI, VII e VIII; "
            "coordena respiração (centro pneumotáxico e apnêustico); "
            "conecta cerebelo ao córtex via pedúnculos cerebelares médios. "
            "Bulbo (medula oblonga): controla frequência cardíaca, pressão arterial, "
            "respiração (centro respiratório dorsal e ventral), deglutição, vômito "
            "e tosse. Nervos IX, X, XI e XII. Ponto de decussação das vias motoras "
            "(pirâmides bulbares) — por isso lesões abaixo da decussação afetam o "
            "mesmo lado, e acima afetam o lado oposto."
        ),

        "clinical_relevance": (
            "Lesões do tronco produzem síndromes cruzadas — característica neurológica "
            "exclusiva desta região: déficit de nervo craniano IPSILATERAL à lesão "
            "(núcleo do nervo está no mesmo lado) + déficit motor/sensorial "
            "CONTRALATERAL (vias cruzaram acima ou abaixo da lesão). "
            "Síndrome de Wallenberg (infarto lateral do bulbo pela PICA): "
            "vertigem, disfagia, soluço, síndrome de Horner ipsilateral, "
            "hemianestesia contralateral ao corpo com anestesia ipsilateral na face. "
            "DIPG (glioma difuso intrínseco da ponte) é o tumor pediátrico com pior "
            "prognóstico em oncologia: mediana de sobrevida de 9-11 meses, "
            "sem ressecção cirúrgica possível pela localização. "
            "Lesões bilaterais do tronco causam coma por disfunção do SRAA "
            "(sistema reticular ativador ascendente)."
        ),

        "pathologies": [
            "DIPG — glioma difuso intrínseco da ponte (predominantemente pediátrico)",
            "Cavernoma pontino — sangramento em 'pipoca' na RM",
            "AVC do tronco — artéria basilar (devastador se oclusão total)",
            "Esclerose múltipla — placas no tronco",
            "Glioma do tronco do adulto — melhor prognóstico que DIPG",
        ],

        "brodmann_areas": ["N/A — estrutura subcortical com núcleos de nervos cranianos III–XII"],

        "vascularization": "Artéria basilar e seus ramos perfurantes; "
                           "PICA (artéria cerebelar posteroinferior) — bulbo lateral; "
                           "AICA — bulbo/ponte lateral; SCA — parte superior",

        "color": "#8E44AD",

        # Estrutura mediana pequena — prioridade alta no sistema de detecção
        "coords": {
            "u_min": 0.33, "u_max": 0.67,
            "v_min": 0.54, "v_max": 0.74,
        },
    },

    # ── Cerebelo ──────────────────────────────────────────────────────────────
    "cerebellum": {
        "name": "Cerebelo",

        "function": (
            "O cerebelo contém mais de 50% de todos os neurônios do sistema nervoso "
            "central, apesar de ocupar apenas ~10% do volume encefálico. "
            "Sua função central é a coordenação motora fina: recebe cópias dos comandos "
            "motores do córtex (via ponte), feedback proprioceptivo dos músculos e "
            "articulações (via medula), e informações do sistema vestibular (equilíbrio). "
            "Compara continuamente o movimento planejado com o executado e envia sinais "
            "corretivos ao córtex motor via tálamo — funciona como um 'controlador preditivo'. "
            "O vermis (porção mediana) coordena equilíbrio e marcha. "
            "Os hemisférios cerebelares coordenam movimentos apendiculares ipsilaterais "
            "(membros superiores e inferiores do mesmo lado). "
            "O cerebelo também participa de funções cognitivas e afetivas "
            "(síndrome cognitivo-afetiva do cerebelo, descrita por Schmahmann)."
        ),

        "clinical_relevance": (
            "Lesões cerebelares produzem síndrome cerebelar caracterizada por: "
            "Ataxia: descoordenação da marcha (marcha em 'bêbado', base alargada); "
            "Dismetria: incapacidade de atingir um alvo com precisão (teste índex-nariz); "
            "Disdiadococinesia: incapacidade de movimentos alternados rápidos; "
            "Tremor de intenção: tremor que piora ao aproximar do alvo; "
            "Nistagmo: oscilação rítmica involuntária dos olhos; "
            "Disartria escandida: fala 'explosiva', silabada, irregular em ritmo. "
            "Meduloblastoma é o tumor cerebral mais comum em crianças "
            "(localização: vermis — bloqueia o 4º ventrículo causando hidrocefalia obstrutiva). "
            "Hemangioblastoma cerebelar é o tumor característico da síndrome de von Hippel-Lindau."
        ),

        "pathologies": [
            "Meduloblastoma — tumor pediátrico mais comum (vermis cerebelar)",
            "Hemangioblastoma — tumor vascularizado, associado a VHL",
            "Ependimoma — origina-se do 4º ventrículo",
            "Metástases — localização frequente (boa vascularização)",
            "AVC cerebelar — PICA, AICA ou SCA",
            "Degeneração cerebelar paraneoplásica",
        ],

        "brodmann_areas": ["N/A — córtex cerebelar com células de Purkinje (não numerado por Brodmann)"],

        "vascularization": "PICA (artéria cerebelar posteroinferior) — hemi inferior; "
                           "AICA (artéria cerebelar anteroinferior) — hemi anteroinferior; "
                           "SCA (artéria cerebelar superior) — hemi superior",

        "color": "#16A085",

        "coords": {
            "u_min": 0.18, "u_max": 0.82,
            "v_min": 0.73, "v_max": 1.00,
        },
    },
}

### --------------------- Priority System --------------###

PRIORITY: list[str] = [
    "corpus_callosum",
    "brainstem",
    "temporal_lobe_left",
    "temporal_lobe_right",
    "frontal_lobe",
    "occipital_lobe",
    "cerebellum",
    "parietal_lobe",
]

### --------------------- Unknown Regions --------------###

_UNKNOWN_REGION: RegionDict = {
    "name": "Região não identificada",
    "function": (
        "Clique em uma região da imagem MRI para obter informações anatômicas "
        "e clínicas detalhadas. As regiões mapeadas incluem: lobo frontal, "
        "lobo parietal, lobos temporais esquerdo e direito, lobo occipital, "
        "corpo caloso, tronco cerebral e cerebelo."
    ),
    "clinical_relevance": (
        "Selecione uma área específica da imagem para ver informações clínicas, "
        "patologias associadas e vascularização da região."
    ),
    "pathologies": [],
    "brodmann_areas": [],
    "vascularization": "N/A",
    "color": "#95A5A6",
    "coords": {},
}

### --------------------- Public Functions --------------###

def get_regions_by_coords(u: float, v: float) -> str:

    u = max(0,0, min(1.0, float(u)))
    v = max(0,0, min(1.0, float(v)))

    for region_id in PRIORITY:
        coords = BRAIN_REGIONS[region_id].get("coords", {})
        if not coords:
            continue

        if coords["u_min"] <= u <= coords["u_max"] and coords["v_min"] <= v <= coords["v_max"]:
            return region_id
    return "unknow"

def get_region_info(region_id: str) -> RegionDict:
    return BRAIN_REGIONS[region_id].get(region_id, _UNKNOWN_REGION)

def get_region_full(u: float, v: float) -> tuple[str, RegionDict]:
    region_id = get_regions_by_coords(u, v)
    return region_id, get_region_info(region_id)

def list_all_regions() -> list[dict[str, str]]:
        ordered_ids = PRIORITY + [
            rid for rid in BRAIN_REGIONS if rid not in PRIORITY
        ]

        return [
            {
                "region_id": rid,
                "name": str(BRAIN_REGIONS[rid]["name"]),
                "color": str(BRAIN_REGIONS[rid]["color"]),
            }
            for rid in ordered_ids
            if rid in BRAIN_REGIONS
        ]

