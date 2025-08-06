#  Grade-Specific Parameter-Efficient Fine-Tuning for Readability Assessment




## Installation ##
Please install the dependencies via conda, using the following command:

```bash
pip install -r requirements.txt
```

## Experiment

### Fine-Tuning with LLaMA-Factory

We utilize the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework for fine-tuning. Please refer to the official repository for detailed implementation instructions.

To launch the Web UI for fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
```

### OPPG: One PEFT Per Grade

#### Fine-Tuning Multi-Level Models

We fine-tune separate models on datasets of different grade levels to obtain:

- `Level-1 LLM`
- `Level-2 LLM`
- `Level-3 LLM`
- `Level-4 LLM`

Each level corresponds to a model adapted for assessing text readability appropriate to a specific grade.


### Evaluation

After fine-tuning, start OpenAI-style API services for each level-specific model:

```bash
CUDA_VISIBLE_DEVICES=0 GRADIO_SERVER_PORT=7860 llamafactory-cli api examples/inference/level_1.yaml
CUDA_VISIBLE_DEVICES=0 GRADIO_SERVER_PORT=7861 llamafactory-cli api examples/inference/level_2.yaml
CUDA_VISIBLE_DEVICES=1 GRADIO_SERVER_PORT=7862 llamafactory-cli api examples/inference/level_3.yaml
CUDA_VISIBLE_DEVICES=1 GRADIO_SERVER_PORT=7863 llamafactory-cli api examples/inference/level_4.yaml
```
Once the services are running, evaluate your dataset using:
```bash
python ./code/OPPG.py \
    --input_file {DATA_PATH} \
    --output_file {RESULT_JSON_PATH}
```

```Finally```, the accuracy, recall, precision, F1, and confusion matrix of the prediction results are calculated.
```bash
python ./code/test.py \
    --input_file {RESULT_JSON_PATH} \
```
