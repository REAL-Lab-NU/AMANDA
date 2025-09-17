# AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering

**🎉 Accepted to EMNLP 2025 Findings!**

## Abstract

Medical Multimodal Large Language Models (Med-MLLMs) have shown great promise in medical visual question answering (MedVQA). However, when deployed in low-resource settings where abundant labeled data are unavailable, existing Med-MLLMs commonly fail due to their medical reasoning capability bottlenecks: (i) the intrinsic reasoning bottleneck that ignores the details from the medical image; (ii) the extrinsic reasoning bottleneck that fails to incorporate specialized medical knowledge. To address those limitations, we propose AMANDA, a training-free agentic framework that performs medical knowledge augmentation via LLM agents. Specifically, our intrinsic medical knowledge augmentation focuses on coarse-to-fine question decomposition for comprehensive diagnosis, while extrinsic medical knowledge augmentation grounds the reasoning process via biomedical knowledge graph retrieval. Extensive experiments across eight Med-VQA benchmarks demonstrate substantial improvements in both zero-shot and few-shot Med-VQA settings.

## 🏗️ Framework Overview

### AMANDA Pipeline Architecture
![AMANDA Framework](figures/pipeline.pdf)

The AMANDA framework comprises five specialized agents working collaboratively:
- **Perceiver**: Generates medical image descriptions and initial answers
- **Reasoner**: Synthesizes information for refined medical reasoning  
- **Evaluator**: Assesses confidence and determines if additional knowledge is needed
- **Explorer**: Performs intrinsic knowledge augmentation through coarse-to-fine question decomposition
- **Retriever**: Provides extrinsic knowledge augmentation via biomedical knowledge graphs

### Adaptive Reasoning and In-Context Learning
![Adaptive Components](figures/icl.pdf)

AMANDA features two key enhancement mechanisms:
- **(a) Adaptive Reasoning Refinement**: Dynamic confidence-based control to balance thoroughness with computational efficiency
- **(b) In-Context Examples Selection**: Dual-similarity selection strategy using both visual and textual embeddings for few-shot learning

## 📂 File Structure

### Core Framework Files
- **`amanda_med_vqa.py`** - Main pipeline script implementing the AMANDA framework for medical VQA tasks
- **`amanda_prompts.py`** - Contains all system prompts and prompt templates for different agents (Perceiver, Reasoner, Evaluator, Explorer, Retriever)

### Model Components
- **`MLLM/`** - Directory containing multimodal large language model implementations
  - `models/` - Model architectures and loading utilities
  - `conversation/` - Conversation templates and dialogue management
  - `processors/` - Image and text preprocessing components
  - `configs/` - Configuration files for different model variants

### Knowledge Retrieval
- **`Retriever/`** - Knowledge augmentation and retrieval components
  - `utility.py` - Utility functions for biomedical knowledge graph queries and retrieval
  - `config_loader.py` - Configuration loader for retrieval settings

## 📖 Citation

```bibtex
@inproceedings{wang2025amanda,
    title={AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering},
    author={Wang, Ziqing and Mao, Chengsheng and Wen, Xiaole and Yuan, Luo and Ding, Kaize},
    booktitle={Findings of the Association for Computational Linguistics: EMNLP 2025},
    year={2025}
}
```