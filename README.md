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

| Subset | Description | Documents | Queries | Example Query | Link |
|--------|-------------|-----------|---------|---------------|------|
| **HR** | Workforce outlook and employment policy | 2,109 | 221 | 산업용 첨단화학소재 분야의 대졸 채용률, 구매·영업·시장조사 직무의 채용률, 생산기술 직무의 채용-퇴직 격차를 비교하여 인력 수급 불균형 원인을 분석하라. | [🤗 Dataset](https://huggingface.co/datasets/whybe-choi/kovidore-v2-hr-beir) |
| **Energy** | Energy policy and power market trends | 1,993 | 173 | 액화석유가스 안전공급 계약제에서 체적판매방법과 중량판매방법의 최소 계약기간은 어떻게 다르며, 소비자보장책임보험의 최대 보상한도는 얼마인가요? | [🤗 Dataset](https://huggingface.co/datasets/whybe-choi/kovidore-v2-energy-beir) |
| **Economic** | Quarterly economic trend reports | 1,477 | 163 | 2022년 원유 도입 단가 상승과 원/달러 환율 변동이 국내 회사채 수익률에 미친 영향을 비교 분석하라 | [🤗 Dataset](https://huggingface.co/datasets/whybe-choi/kovidore-v2-economic-beir) |
| **Cybersecurity** | Cyber threat analysis and security guides | 1,150 | 149 | 네트워크 백업의 보안 취약점을 해결하기 위해 WORM 스토리지 기술이 어떻게 적용되는가? | [🤗 Dataset](https://huggingface.co/datasets/whybe-choi/kovidore-v2-cybersecurity-beir) |

## Performance Leaderboard

### KoViDoRe v1
The following table shows performance across all KoViDoRe v1 tasks (ndcg@5 scores as percentages, sorted by Average):

| Model | Model Size | FinOCR | MIR | Office | Slide | VQA | Average |
|-------|------------|--------|-----|--------|-------|-----|---------|
| **jinaai/jina-embeddings-v4** | 3800 | 94.1 | 73.6 | 88.7 | 89.7 | 86.3 | 86.5 |
| **TomoroAI/tomoro-colqwen3-embed-8b** | 8000 | 81.8 | 60.9 | 84.2 | 86.3 | 82.9 | 79.2 |
| **nomic-ai/colnomic-embed-multimodal-7b** | 7000 | 78.0 | 63.4 | 82.0 | 86.8 | 85.2 | 79.1 |
| **ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1** | 7000 | 65.9 | 60.2 | 79.5 | 84.2 | 82.1 | 74.4 |
| **nomic-ai/colnomic-embed-multimodal-3b** | 3000 | 75.5 | 56.3 | 82.2 | 36.3 | 72.9 | 64.6 |
| **vidore/colqwen2-v1.0** | 2210 | 61.6 | 44.0 | 56.7 | 66.0 | 67.5 | 59.2 |
| **TomoroAI/tomoro-colqwen3-embed-4b** | 4000 | 67.1 | 32.4 | 42.5 | 66.9 | 52.8 | 52.3 |
| **vidore/colqwen2.5-v0.2** | 3000 | 45.0 | 48.0 | 62.2 | 25.6 | 68.0 | 49.8 |
| **ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-3B-v1** | 3000 | 43.0 | 37.5 | 53.7 | 24.8 | 62.3 | 44.3 |
| **eagerworks/eager-embed-v1** | 4000 | 17.9 | 20.8 | 37.0 | 60.6 | 49.0 | 37.1 |
| **vidore/colpali-v1.3** | 2920 | 38.2 | 14.9 | 23.6 | 50.7 | 30.0 | 31.5 |
| **vidore/colpali-v1.2** | 2920 | 37.1 | 13.2 | 24.8 | 46.7 | 28.4 | 30.0 |
| **vidore/colpali-v1.1** | 2920 | 35.4 | 16.4 | 19.0 | 44.1 | 25.6 | 28.1 |
| **vidore/colSmol-500M** | 500 | 43.6 | 3.7 | 7.4 | 13.5 | 6.2 | 14.9 |
| **jinaai/jina-clip-v2** | 865 | 1.1 | 8.4 | 14.4 | 33.3 | 11.6 | 13.8 |
| **vidore/colSmol-256M** | 256 | 37.4 | 3.2 | 4.8 | 10.8 | 5.6 | 12.4 |
| **google/siglip-so400m-patch14-384** | 878 | 4.0 | 3.9 | 6.3 | 21.3 | 7.2 | 8.5 |
| **TIGER-Lab/VLM2Vec-Full** | 4150 | 1.7 | 1.6 | 8.0 | 15.0 | 6.7 | 6.6 |
| **laion/CLIP-ViT-bigG-14-laion2B-39B-b160k** | 2540 | 0.5 | 1.9 | 3.3 | 12.5 | 5.6 | 4.8 |
| **openai/clip-vit-base-patch16** | 151 | 0.3 | 0.6 | 0.0 | 5.9 | 3.3 | 2.0 |

### KoViDoRe v2
The following table shows performance across all KoViDoRe v2 tasks (ndcg@10 scores as percentages, sorted by Average):

| Model | Model Size | Cybersecurity | Economic | Energy | HR | Average |
|-------|------------|---------------|----------|--------|-----|---------|
| **jinaai/jina-embeddings-v4** | 3800 | 77.6 | 24.5 | 67.7 | 50.1 | 55.0 |
| **TomoroAI/tomoro-colqwen3-embed-8b** | 8000 | 73.7 | 16.3 | 58.5 | 26.5 | 43.8 |
| **nomic-ai/colnomic-embed-multimodal-7b** | 7000 | 69.6 | 12.4 | 59.5 | 33.3 | 43.7 |
| **ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1** | 7000 | 66.0 | 12.1 | 55.4 | 26.4 | 40.0 |
| **nomic-ai/colnomic-embed-multimodal-3b** | 3000 | 47.4 | 10.5 | 44.2 | 32.9 | 33.8 |
| **vidore/colqwen2-v1.0** | 2210 | 53.3 | 8.0 | 42.0 | 14.7 | 29.5 |
| **TomoroAI/tomoro-colqwen3-embed-4b** | 4000 | 55.3 | 9.1 | 31.0 | 10.1 | 26.4 |
| **vidore/colqwen2.5-v0.2** | 3000 | 43.9 | 3.9 | 44.3 | 13.5 | 26.4 |
| **eagerworks/eager-embed-v1** | 4000 | 51.5 | 5.4 | 32.7 | 7.0 | 24.2 |
| **ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-3B-v1** | 3000 | 41.4 | 6.3 | 31.5 | 11.3 | 22.6 |
| **vidore/colpali-v1.3** | 2920 | 34.7 | 1.6 | 20.6 | 6.2 | 15.8 |
| **vidore/colpali-v1.1** | 2920 | 31.9 | 3.0 | 18.2 | 6.0 | 14.8 |
| **vidore/colpali-v1.2** | 2920 | 33.2 | 2.1 | 16.4 | 4.5 | 14.1 |
| **vidore/colSmol-500M** | 500 | 26.2 | 0.6 | 9.9 | 0.9 | 9.4 |
| **jinaai/jina-clip-v2** | 865 | 20.4 | 0.2 | 11.3 | 3.1 | 8.8 |
| **vidore/colSmol-256M** | 256 | 19.7 | 1.0 | 9.5 | 1.1 | 7.8 |
| **google/siglip-so400m-patch14-384** | 878 | 15.3 | 1.3 | 5.3 | 1.1 | 5.8 |
| **laion/CLIP-ViT-bigG-14-laion2B-39B-b160k** | 2540 | 13.8 | 0.3 | 4.2 | 0.4 | 4.7 |
| **TIGER-Lab/VLM2Vec-Full** | 4150 | 9.8 | 1.3 | 3.2 | 1.3 | 3.9 |
| **openai/clip-vit-base-patch16** | 151 | 4.1 | 0.0 | 0.8 | 0.6 | 1.4 |

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
```
