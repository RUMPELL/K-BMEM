<p align="center">
  <img src="assets/kbmem-banner.svg" alt="K-BMEM — 한영 의료 검색 임베딩" width="100%">
</p>

<p align="center"><a href="README.md">English</a> · <a href="#핵심-결과">핵심 결과</a> · <a href="#빠른-실행">빠른 실행</a></p>

# K-BMEM

K-BMEM은 개인정보 보호가 중요한 폐쇄망 환경을 목표로 한 **한–영 의료 dense retrieval 연구 프로토타입**입니다.

한국 의료 텍스트에는 한국어 설명과 영문 질환명·약물명·검사명·약어가 빈번히 섞입니다. 이 프로젝트는 이런 환경에서 의료 의미를 구별하는 로컬 임베딩 모델을 만들고, 불확실성을 포함한 재현 가능한 평가로 검증하는 과정을 다룹니다.

> 연구 및 포트폴리오 목적의 프로토타입입니다. 의료기기나 임상 의사결정 지원 도구가 아니며, 모델 가중치와 원본 데이터셋은 이 저장소에서 배포하지 않습니다.

## 핵심 기여

| 영역 | 구현 내용 |
|---|---|
| 데이터 | 길이 편향과 in-batch false negative를 통제한 대조학습 배치 |
| 학습 | 결정론적 batch plan과 checkpoint identity를 갖춘 로컬 fine-tuning |
| 평가 | paired bootstrap CI, exact McNemar, sparse/dense/상용 API 비교 |
| 실험 | DAPT, reranker distillation, hybrid retrieval, code-switching, 데이터 소스 ablation |
| 배포 | 추론 시 외부 API가 필요 없는 on-premise 검색 구조 |

## 핵심 결과

| 모델 | 의료 Exam accuracy@1 ↑ | AIHub paired passage nDCG@10 ↑ |
|---|---:|---:|
| **K-BMEM** | 0.2519 | **0.9061** |
| KURE | 0.2331 | 0.8246 |
| BGE-M3 | 0.2218 | 0.8195 |
| Qwen3-Embedding-0.6B | 0.2556 | 0.8847 |
| OpenAI text-embedding-3-small | **0.2744** | 0.5079 |
| OpenAI text-embedding-3-large | 0.2707 | 0.7780 |

- Exam은 한국어 의료 객관식 266문항의 보기 순위 평가입니다. 여섯 모델 간 차이는 Holm 보정 후 통계적으로 유의하지 않았습니다.
- AIHub는 질문–정답 문단 339쌍의 짝 맞추기 평가입니다. K-BMEM은 KURE, BGE-M3, 두 OpenAI 모델보다 Holm 보정 후 높았고 Qwen3와의 차이는 유의하지 않았습니다.
- AIHub는 실제 open-corpus RAG가 아닙니다. 두 평가 split 모두 과거 노출 이력이 있어 pristine final test로 주장하지 않습니다.

## 빠른 실행

```bash
git clone https://github.com/RUMPELL/K-BMEM.git
cd K-BMEM
python -m venv .venv
source .venv/bin/activate
pip install -e .

kbmem-search \
  --model nlpai-lab/KURE-v1 \
  --query "갑자기 발생한 국소 신경학적 증상"
```

예제 corpus는 공개를 위해 새로 작성한 합성 텍스트입니다. 폐쇄망에서는 Hugging Face ID 대신 로컬 모델 경로를 전달하세요.

## 연구 결과를 해석하는 방식

K-BMEM은 모든 평가에서 항상 1등인 모델이 아닙니다. DAPT, 두 종류의 지식 증류, 확장 데이터 학습, sparse+dense hybrid, Qwen fine-tuning 등은 사전에 고정한 기준을 만족하지 못해 채택하지 않았습니다. 이 저장소는 성공한 숫자뿐 아니라 어떤 실험을 왜 기각했는지도 연구 성과로 봅니다.

## 공개 범위

- 포함: 범용 검색 코드, 배치 구성·학습·평가 파이프라인 원본 코드와 테스트([`research/`](research/README.md)), 합성 예제, 모델 단위 집계 결과, 모델·데이터 카드
- 제외: 모델 가중치, 원본/가공 데이터, 행 단위 결과, query ID/qrel, embedding/index/cache

코드는 [MIT License](LICENSE)로 공개합니다. 제3자 모델과 데이터에는 각각의 원 라이선스가 적용됩니다.
