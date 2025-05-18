# Chinese Text Readability Assessment via Grade-Specific Parameter-Efficient Fine-tuning


This is source code of our EMNLP 2025 paper


## Overview

- **Interpretability & Educational Alignment**:  
  In educational NLP, assessing the readability of Chinese texts is crucial yet challenging. While deep learning methods have demonstrated strong performance due to their expressive power, they often suffer from limited interpretability. This makes it difficult to understand or trust the model's decisions, particularly in cognitively sensitive educational settings. Furthermore, existing models struggle with fine-grained comprehension distinctions across different grade levels, limiting their effectiveness in real-world classroom applications.

- **Grade-specific Language Understanding**:  
  Inspired by the cognitive progression of language acquisition in students, we introduce a novel paradigm that leverages parameter-efficient fine-tuning (PEFT) to model grade-specific understanding. Our method assigns a dedicated PEFT module to each readability grade, allowing the model to retain and reuse distinct linguistic and cognitive patterns specific to each level. This approach results in a suite of lightweight student models that better simulate how learners at different stages comprehend language.

<div align="center">
  <img src="./asset/PEFT.png" width="50%" height="100%">
</div>

To address the limitations of current methods, we propose **One PEFT Per Grade (OPPG)**, a cognitively inspired framework designed for interpretable and accurate Chinese text readability assessment. OPPG integrates grade-level PEFT modules into a shared pre-trained backbone, combining the strengths of parametric knowledge (captured in grade-specific adapters) with the generalization power of foundation models. By doing so, OPPG enables the creation of interpretable, modular, and scalable student simulators for educational use.

Extensive experiments on a large-scale Chinese textbook corpus demonstrate that OPPG significantly outperforms prompt-based baselines in both classification accuracy and alignment with human-annotated grade labels. In addition, OPPG provides improved transparency in model reasoning, offering clearer explanations aligned with educational theory and making it better suited for deployment in teaching and assessment systems.

<div align="center">
  <img src="./asset/PredDecision.png.png" width="70%" height="100%">
</div>


## Dataset ##

We use publicly available data from the [LaMP](https://arxiv.org/abs/2304.11406) benchmark. You can download the our processed data [here](https://drive.google.com/file/d/10MR_FsAhm8rpYRra9jTc8qq22DfjXG-A/view?usp=sharing), unzip it, and place it under the ```./data``` folder


## Installation ##
Please install the dependencies via conda, using the following command:

```bash
pip install -r requirements.txt
```

## Experiment ##
```task_name``` can be selected from ```[citation, movie_tagging, news_categorize, news_headline, product_rating, scholarly_title, tweet_paraphrase]```. Here, we take ```movie_tagging``` as an example.

### OPPU
#### 1. Base LLM Task Adaption

```bash
CUDA_VISIBLE_DEVICES=0 python task_LoRA.py --k 0 --task_name movie_tagging
```

#### 2. Train One PEFT Per User
```bash
CUDA_VISIBLE_DEVICES=0 python OPPU.py --k 0 --task_name movie_tagging --task_lora ./ckpt/movie_tagging/k0-movie_tagging-Llama-2-7b-hf-task_LoRA_ckpt
```

### OPPU + RAG

#### 1. Base LLM Task Adaption

```bash
CUDA_VISIBLE_DEVICES=0 python task_LoRA.py --k 1 --task_name movie_tagging
```

#### 2. Train One PEFT Per User
```bash
CUDA_VISIBLE_DEVICES=0 python OPPU.py --k 1 --task_name movie_tagging --task_lora ./ckpt/movie_tagging/k1-movie_tagging-Llama-2-7b-hf-task_LoRA_ckpt
```
----

### OPPU + PAG
#### 1. Base LLM Task Adaption

```bash
CUDA_VISIBLE_DEVICES=0 python task_LoRA.py --k 1 --task_name movie_tagging --add_profile
```

#### 2. Train One PEFT Per User
```bash
CUDA_VISIBLE_DEVICES=0 python OPPU.py --k 1 --task_name movie_tagging --task_lora ./ckpt/movie_tagging/k1-movie_tagging-Llama-2-7b-hf-profile-task_LoRA_ckpt --add_profile
```

## Evaluation ##
```TASK_ID``` is the corresponding ID selected from ```["LaMP_1", "LaMP_2N", "LaMP_2M", "LaMP_3", "LaMP_4", "LaMP_5", "LaMP_7"]```

```bash
python ./eval/eval_task.py \
    --golds_json {PATH_TO_LABEL_JSON_FILE} \
    --preds_json {PATH_TO_PREDICTION_JSON_FILE} \
    --task_name {TASK_ID} \
    --output_file {RESULT_JSON_PATH}
```

