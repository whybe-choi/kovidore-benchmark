# KoViDoRe Benchmark

Korean Vision Document Retrieval (KoViDoRe) benchmark for evaluating text-to-image retrieval models on Korean visual documents.

## Overview
<img src="assets/cover.png" alt="cover">

**KoViDoRe** is a comprehensive benchmark for evaluating Korean visual document retrieval capabilities. Built upon the foundation of [ViDoRe](https://huggingface.co/vidore), it assesses how well models can retrieve relevant Korean visual documents—including screenshots, presentation slides, and office documents—when given Korean text queries.

The **KoViDoRe v1** encompasses 5 distinct tasks, each targeting different types of visual documents commonly found in Korean business and academic environments. This diverse task structure allows for thorough evaluation of multimodal retrieval performance across various document formats and content types.

The **KoViDoRe v2** addresses a key limitation of KoViDoRe v1—single-page matching—by generating queries that require aggregating information across multiple pages. This benchmark consists of 4 distinct tasks targeting practical enterprise domains: cybersecurity, economic reports, energy documents, and HR materials.


## Tasks & Examples

### KoViDoRe v1

| Task | Description | Documents | Queries | Example Query | Sample Image |
|------|-------------|-----------|---------|---------------|--------------|
| **MIR** | Multimodal Information Retrieval | 1,366 | 1,496 | 코로나19 동절기 집중접종기간 운영개요 혼합형에 대해 알려주세요 | <img src="assets/examples/mir_sample.jpg" width="200" alt="MIR"> |
| **VQA** | Visual Question Answering | 1,101 | 1,500 | 경제협력 품목 중 가장 적은 교역액과 가장 많은 교역액의 차이는 얼마인가요? | <img src="assets/examples/vqa_sample.png" width="200" alt="VQA"> |
| **Slide** | Presentation Slides | 1,415 | 180 | 포털 사이트나 콘텐츠 제공자가 기존 콘텐츠를 다양한 장치로 서비스할 때 얻는 이점은 무엇인가? | <img src="assets/examples/slide_sample.jpg" width="200" alt="Slide"> |
| **Office** | Office Documents | 1,993 | 222 | 정치·사회 이슈를 주제로 하는 유튜브 채널을 통해 정보를 얻는 비율은 얼마인가요? | <img src="assets/examples/office_sample.jpg" width="200" alt="Office"> |
| **FinOCR** | Financial OCR Documents | 2,000 | 198 | 반려동물보험에 가입한 보험계약자 공형진의 증권번호는 무엇인가요? | <img src="assets/examples/finocr_sample.png" width="200" alt="FinOCR"> |

### KoViDoRe v2

| Subset | Description | Documents | Queries | Link |
|--------|-------------|-----------|---------|------|
| **HR** | Workforce outlook and employment policy | 2,109 | 221 | [🤗 Dataset](https://huggingface.co/datasets/whybe-choi/kovidore-v2-hr-beir) |
| **Energy** | Energy policy and power market trends | 1,911 | 190 | [🤗 Dataset](https://huggingface.co/datasets/whybe-choi/kovidore-v2-energy-beir) |
| **Economic** | Quarterly economic trend reports | 1,477 | 163 | [🤗 Dataset](https://huggingface.co/datasets/whybe-choi/kovidore-v2-economic-beir) |
| **Cybersecurity** | Cyber threat analysis and security guides | 1,150 | 149 | [🤗 Dataset](https://huggingface.co/datasets/whybe-choi/kovidore-v2-cybersecurity-beir) |

## Performance Leaderboard

### KoViDoRe v1
The following table shows performance across all KoViDoRe v1 tasks (ndcg@5 scores as percentages):

| Model | Model Size | FinOCR | MIR | Office | Slide | VQA | Average | ViDoRe V2 (Eng) |
|-------|------------|--------|-----|--------|-------|-----|---------|------------------|
| **nomic-ai/colnomic-embed-multimodal-3b** | 3000 | 82.2 | 70.7 | 86.3 | 78.4 | 84.4 | 80.4 | 55.5 |
| **nomic-ai/colnomic-embed-multimodal-7b** | 7000 | 81.9 | 67.9 | 85.9 | 87.6 | 87.2 | 82.1 | 60.8 |
| **vidore/colqwen2.5-v0.2** | 3000 | 67.3 | 62.5 | 75.3 | 78.0 | 81.0 | 72.8 | 59.3 |
| **vidore/colqwen2-v1.0** | 2210 | 66.3 | 57.4 | 68.7 | 73.9 | 75.5 | 68.4 | 55.0 |
| **jinaai/jina-embeddings-v4** | 3800 | 88.9 | 73.8 | 88.6 | 89.5 | 86.2 | 85.4 | 57.6 |
| **vidore/colpali-v1.2** | 2920 | 43.8 | 20.2 | 28.4 | 51.2 | 36.8 | 36.1 | 50.7 |
| **vidore/colpali-v1.3** | 2920 | 42.6 | 18.8 | 26.4 | 55.3 | 36.6 | 35.9 | 54.2 |
| **vidore/colpali-v1.1** | 2920 | 38.3 | 19.0 | 25.3 | 48.6 | 30.0 | 32.2 | 47.2 |
| **nvidia/llama-nemoretriever-colembed-3b-v1** | 3000 | TBA | TBA | TBA | TBA | TBA | TBA | 63.5 |
| **nvidia/llama-nemoretriever-colembed-1b-v1** | 1000 | 76.6 | 28.1 | 34.2 | 53.3 | 39.4 | 46.3 | 62.1 |
| **vidore/colSmol-500M** | 500 | 50.9 | 4.7 | 9.7 | 16.1 | 7.4 | 17.8 | 43.5 |
| **vidore/colSmol-256M** | 256 | 46.6 | 4.0 | 8.4 | 13.9 | 7.6 | 16.1 | 32.9 |
| **google/siglip-so400m-patch14-384** | 878 | 4.0 | 3.9 | 6.3 | 21.3 | 7.3 | 8.6 | 31.4 |
| **TIGER-Lab/VLM2Vec-Full** | 4150 | 1.4 | 1.6 | 7.2 | 14.9 | 6.8 | 6.4 | 30.1 |
| **laion/CLIP-ViT-bigG-14-laion2B-39B-b160k** | 2540 | 0.5 | 1.9 | 3.7 | 12.5 | 5.6 | 4.8 | 17.6 |
| **openai/clip-vit-base-patch16** | 151 | 0.3 | 0.6 | 0.0 | 5.9 | 3.3 | 2.5 | 8.3 |
| **ibm-granite/granite-vision-3.3-2b-embedding** | 2980 | 0.0 | 0.4 | 0.6 | 0.3 | 0.0 | 0.26 | 58.1 |

### KoViDoRe v2
The following table shows performance across all KoViDoRe v2 tasks (ndcg@10 scores as percentages):

| Model | Model Size | Cybersecurity | Economic | Energy | HR | Average | KoViDoRe V1 (Kor) |
|-------|------------|---------------|----------|--------|-----|---------|------------------|
| **nomic-ai/colnomic-embed-multimodal-3b** | 3000 | 73.7 | 17.8 | 61.0 | 37.0 | 47.4 | 80.4 |
| **nomic-ai/colnomic-embed-multimodal-7b** | 7000 | 72.3 | 19.9 | 56.7 | 35.8 | 46.2 | 82.1 |
| **vidore/colqwen2.5-v0.2** | 3000 | 60.8 | 12.6 | 48.1 | 22.9 | 36.1 | 72.8 |
| **vidore/colqwen2-v1.0** | 2210 | 59.9 | 10.4 | 37.7 | 23.8 | 33.0 | 68.4 |
| **jinaai/jina-embeddings-v4** | 3800 | 77.3 | 25.5 | 61.7 | 50.4 | 53.7 | 85.4 |
| **vidore/colpali-v1.2** | 2920 | 40.9 | 2.0 | 18.2 | 5.9 | 16.8 | 36.1 |
| **vidore/colpali-v1.3** | 2920 | 37.8 | 1.7 | 17.8 | 7.0 | 16.1 | 35.9 |
| **vidore/colpali-v1.1** | 2920 | 35.6 | 2.7 | 17.7 | 6.5 | 15.6 | 32.2 |
| **nvidia/llama-nemoretriever-colembed-3b-v1** | 3000 | TBA | TBA | TBA | TBA | TBA | TBA |
| **nvidia/llama-nemoretriever-colembed-1b-v1** | 2418 | 52.1 | TBA | TBA | TBA | TBA | 46.3 |
| **vidore/colSmol-500M** | 500 | 27.0 | 1.1 | 6.3 | 1.2 | 8.9 | 17.8 |
| **vidore/colSmol-256M** | 256 | 23.1 | 1.1 | 5.7 | 1.3 | 7.8 | 16.1 |
| **google/siglip-so400m-patch14-384** | 878 | 15.3 | 1.3 | 3.3 | 1.1 | 5.3 | 8.6 |
| **TIGER-Lab/VLM2Vec-Full** | 4150 | 9.8 | 1.3 | 2.8 | 1.2 | 3.8 | 6.4 |
| **laion/CLIP-ViT-bigG-14-laion2B-39B-b160k** | 2540 | 13.7 | 0.3 | 2.4 | 0.4 | 4.2 | 4.8 |
| **openai/clip-vit-base-patch16** | 151 | 4.1 | 0.0 | 0.7 | 0.6 | 1.3 | 2.5 |
| **ibm-granite/granite-vision-3.3-2b-embedding** | 2980 | 0.0 | 0.5 | 0.3 | 0.4 | 0.3 | 0.3 |

## Interpretability

We provide interpretability maps to help understand how different models attend to document image patches when processing queries. Each row in the tables represents interpretability maps for different query words.

- Query: **인천 광역시의 CT 설치 비율은 몇 프로니?**

| vidore/colpali-v1.3 | vidore/colqwen2.5-v0.2 | jinaai/jina-embeddings-v4 |
|---------------------|------------------------|---------------------------|
| <img src="assets/interpretability/colpali-v1.3/vqa/73209/03_설치.png" width="500" alt="interpretability"> | <img src="assets/interpretability/colqwen2.5-v0.2/vqa/73209/03_설치.png" width="500" alt="interpretability"> | <img src="assets/interpretability/jina-embeddings-v4/vqa/73209/03_설치.png" width="500" alt="interpretability"> |
| <img src="assets/interpretability/colpali-v1.3/vqa/73209/04_비율은.png" width="500" alt="interpretability"> | <img src="assets/interpretability/colqwen2.5-v0.2/vqa/73209/04_비율은.png" width="500" alt="interpretability"> | <img src="assets/interpretability/jina-embeddings-v4/vqa/73209/04_비율은.png" width="500" alt="interpretability"> |

- Query: **지방자치단체가 보건복지부에 제출하는 문서는 무엇인가요?**

| vidore/colpali-v1.3 | vidore/colqwen2.5-v0.2 | jinaai/jina-embeddings-v4 |
|---------------------|------------------------|---------------------------|
| <img src="assets/interpretability/colpali-v1.3/vqa/69302/01_지방자치단체가.png" width="500" alt="interpretability"> | <img src="assets/interpretability/colqwen2.5-v0.2/vqa/69302/01_지방자치단체가.png" width="500" alt="interpretability"> | <img src="assets/interpretability/jina-embeddings-v4/vqa/69302/01_지방자치단체가.png" width="500" alt="interpretability"> |
| <img src="assets/interpretability/colpali-v1.3/vqa/69302/02_보건복지부에.png" width="500" alt="interpretability"> | <img src="assets/interpretability/colqwen2.5-v0.2/vqa/69302/02_보건복지부에.png" width="500" alt="interpretability"> | <img src="assets/interpretability/jina-embeddings-v4/vqa/69302/02_보건복지부에.png" width="500" alt="interpretability"> |

- Query: **나무가 주거 공간에서 제공하는 역할은 무엇인가?**

| vidore/colpali-v1.3 | vidore/colqwen2.5-v0.2 | jinaai/jina-embeddings-v4 |
|---------------------|------------------------|---------------------------|
| <img src="assets/interpretability/colpali-v1.3/slide/1/01_나무가.png" width="500" alt="interpretability"> | <img src="assets/interpretability/colqwen2.5-v0.2/slide/1/01_나무가.png" width="500" alt="interpretability"> | <img src="assets/interpretability/jina-embeddings-v4/slide/1/01_나무가.png" width="500" alt="interpretability"> |
| <img src="assets/interpretability/colpali-v1.3/slide/1/02_주거.png" width="500" alt="interpretability"> | <img src="assets/interpretability/colqwen2.5-v0.2/slide/1/02_주거.png" width="500" alt="interpretability"> | <img src="assets/interpretability/jina-embeddings-v4/slide/1/02_주거.png" width="500" alt="interpretability"> |

## Installation

```bash
# Install dependencies
uv sync
```

## Quick Start

### Using the CLI

```bash

# Run with custom model
uv run kovidore --model "your-model-name"

# Run specific tasks
uv run kovidore --model "your-model-name" --tasks mir vqa

# Run with custom batch size (default: 16)
uv run kovidore --model "your-model-name" --batch-size 32

# List available tasks
uv run kovidore --list-tasks
```

### Using as a Library

```python
from src.evaluate import run_benchmark

# Run all tasks
evaluation = run_benchmark("your-model-name")

# Run specific tasks
evaluation = run_benchmark("your-model-name", tasks=["mir", "vqa"])

# Run with custom batch size
evaluation = run_benchmark("your-model-name", batch_size=32)
```

## Datasets

> [!Note]
> Unlike KoViDoRe v1, KoViDoRe v2 is freely available on Hugging Face. You can access the full dataset collection [here](https://huggingface.co/collections/whybe-choi/kovidore-benchmark-beir-v2).

We provide pre-processed queries and query-corpus mappings for each task. However, due to licensing restrictions, you'll need to download the image datasets manually from AI Hub (see Acknowledgements section for dataset links).

**Setup Instructions:**
1. Download the required datasets from AI Hub
2. Extract and place images in the following directory structure:
    ```
    data/
    ├── mir/images/
    ├── vqa/images/
    ├── slide/images/
    ├── office/images/
    └── finocr/images/
    ```

The benchmark will automatically locate and use the images from these directories during evaluation.

## Results

Results are automatically saved in the `results/` directory after evaluation completion. The KoViDoRe v1 uses NDCG@5 and the KoViDoRe v2 uses NDCG@10 as the main evaluation metric for all tasks.

## Acknowledgements

This benchmark is inspired by the [ViDoRe](https://huggingface.co/vidore) benchmark. We thank the original authors for their foundational work that helped shape our approach to Korean visual document retrieval.

We also acknowledge the following Korean datasets from AI Hub that were used to construct each task in KoViDoRe v1:

- **[멀티모달 정보검색 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71813)** - Used for KoVidoreMIRRetrieval task
- **[시각화 자료 질의응답 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71812)** - Used for KoVidoreVQARetrieval task  
- **[오피스 문서 생성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71811)** - Used for KoVidoreSlideRetrieval and KoVidoreOfficeRetrieval tasks
- **[OCR 데이터(금융 및 물류)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71301)** - Used for KoVidoreFinOCRRetrieval task

## Contact

For questions or suggestions, please open an issue on the GitHub repository or contact the maintainers:

- [Yongbin Choi](https://github.com/whybe-choi) - whybe.choi@gmail.com
- [Yongwoo Song](https://github.com/facerain) - syw5141@khu.ac.kr

## Citation 

If you use KoViDoRe in your research, please cite as follows:
```bibtex
@misc{KoViDoRe2025,
  author = {Yongbin Choi and Yongwoo Song},
  title = {KoViDoRe: Korean Vision Document Retrieval Benchmark},
  year = {2025},
  url = {https://github.com/whybe-choi/kovidore-benchmark},
  note = {A comprehensive benchmark for evaluating visual document retrieval models on Korean document images}
}

@misc{choi2026kovidorev2,
  author = {Yongbin Choi},
  title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
  year = {2026},
  url = {https://github.com/whybe-choi/kovidore-data-generator},
  note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains}
}
```
