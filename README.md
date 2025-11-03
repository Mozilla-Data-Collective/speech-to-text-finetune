<p align="center">
  <picture>
    <!-- When the user prefers dark mode, show the white logo -->
    <source media="(prefers-color-scheme: dark)" srcset="./images/Blueprint-logo-white.png">
    <!-- When the user prefers light mode, show the black logo -->
    <source media="(prefers-color-scheme: light)" srcset="./images/Blueprint-logo-black.png">
    <!-- Fallback: default to the black logo -->
    <img src="./images/Blueprint-logo-black.png" width="35%" alt="Project logo"/>
  </picture>
</p>


<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%F0%9F%A4%97-yellow)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-%F0%9F%8E%A8-green)](https://www.gradio.app/)
[![Common Voice](https://img.shields.io/badge/Common%20Voice-%F0%9F%8E%A4-orange)](https://commonvoice.mozilla.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr) <br>
[![Docs](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/docs.yaml/)
[![Tests](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/tests.yaml/badge.svg)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/tests.yaml/)
[![Ruff](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/speech-to-text-finetune/actions/workflows/lint.yaml/)

[Blueprints Hub](https://developer-hub.mozilla.ai/)
| [Documentation](https://mozilla-ai.github.io/speech-to-text-finetune/)
| [Getting Started](https://mozilla-ai.github.io/speech-to-text-finetune/getting-started)
| [Contributing](CONTRIBUTING.md)

</div>

# Finetuning Speech-to-Text models: a Blueprint by Mozilla.ai for building your own STT/ASR dataset & model

This blueprint enables you to create your own [Speech-to-Text](https://en.wikipedia.org/wiki/Speech_recognition) dataset and model, optimizing performance for your specific language and use case. Everything runs locally—even on your laptop, ensuring your data stays private. You can finetune a model using your own data or leverage the Common Voice dataset, which supports a wide range of languages. To see the full list of supported languages, visit the [CommonVoice website](https://commonvoice.mozilla.org/en/languages).

<img src="./images/speech-to-text-finetune-diagram.png" width="1200" alt="speech-to-text-finetune Diagram" />


## Example result on Galician

Input Speech audio:


https://github.com/user-attachments/assets/960f1b4f-04b9-4b8d-b988-d51504401e9a

Text output:

| Ground Truth | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | [mozilla-ai/whisper-small-gl](https://huggingface.co/mozilla-ai/whisper-small-gl) *|
| -------------| -------------| ------------------- |
| O Comité Económico e Social Europeo deu luz verde esta terza feira ao uso de galego, euskera e catalán nas súas sesións plenarias, segundo informou o Ministerio de Asuntos Exteriores nun comunicado no que se felicitou da decisión. | O Comité Económico Social Europeo de Uluz Verde está terza feira a Ousse de Gallego e Uskera e Catalan a súas asesións planarias, segundo informou o Ministerio de Asuntos Exteriores nun comunicado no que se felicitou da decisión. | O Comité Económico Social Europeo deu luz verde esta terza feira ao uso de galego e usquera e catalán nas súas sesións planarias, segundo informou o Ministerio de Asuntos Exteriores nun comunicado no que se felicitou da decisión. |

\* Finetuned on the Galician set Common Voice 17.0

👀 You can find a list of finetuned models, created by this Blueprint, on our HuggingFace [collection](https://huggingface.co/collections/mozilla-ai/common-voice-whisper-67b847a74ad7561781aa10fd).

## Quick-start

<div style="text-align: center;">

| Finetune a STT model on Google Colab | Transcribe using a HuggingFace model | Explore all the functionality on GitHub Codespaces|
|----------------------------------------|---------------------------------------------|---------------------------------------------------|
| [![Try Finetuning on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mozilla-ai/speech-to-text-finetune/blob/main/demo/notebook.ipynb) | [![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/mozilla-ai/transcribe) | [![Try on Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=mozilla-ai/speech-to-text-finetune&skip_quickstart=true&machine=standardLinux32gb) |

</div>

## Try it locally

The same instructions apply for the GitHub Codespaces option.

### Setup

1. Create a virtual environment and install dependencies:
   - `pip install -e .`
   - Install [ffmpeg](https://ffmpeg.org) \[Ubuntu\]: `sudo apt install ffmpeg`, \[Mac\]: `brew install ffmpeg`
2. Set up MDC access for Common Voice via the Python SDK:
   - Create an account and get an API key from: https://datacollective.mozillafoundation.org/api-reference
   - Create a local `.env` file from the template and add your MDC API key:
     - `cp example_data/.env.example src/speech_to_text_finetune/.env`
     - Edit `.env` and set `MDC_API_KEY=<your_api_key>`
     - Get an API key from: https://datacollective.mozillafoundation.org/api-reference
3. \[Optional\] Log in to Hugging Face if you plan to track your models: `huggingface-cli login`

### Evaluate existing STT models from the HuggingFace repository

1. Run: `python demo/transcribe_app.py`
2. Add the HF model id of your choice
3. Record a sample of your voice and get the transcribed text back

### Making your own STT model using Custom Data

1. Create your own, local, custom dataset and follow the UI instructions: `python src/speech_to_text_finetune/make_custom_dataset_app.py`
2. Configure `config.yaml` with the model, custom data directory and hyperparameters. If `push_to_hub: True`, log in to HF locally.
3. Finetune: `python src/speech_to_text_finetune/finetune_whisper.py`
4. Test the finetuned model: `python demo/transcribe_app.py`

### Making your own STT model using Common Voice

You can either load Common Voice via the Mozilla Data Collective Python SDK directly or use a locally downloaded copy.

#### Option A: Load via Mozilla Data Collective Python SDK

1. Ensure `.env` exists and contains a valid `MDC_API_KEY` (see Setup above).
2. Identify the MDC dataset id for your language (Scripted or Spontaneous Common Voice) from the Mozilla Data Collective portal.
    - You can find the `id` by looking at the URL of the dataset's page on MDC platform. The ID is located at the very end of the URL, after the `/datasets/` path. For example, for URL `https://datacollective.mozillafoundation.org/datasets/cmflnuzw6lrt9e6ui4kwcshvn` dataset id will be `cmflnuzw6lrt9e6ui4kwcshvn`.
3. Set `dataset_id` in `config.yaml` to the MDC dataset id. Example:
   ```
   model_id: openai/whisper-tiny
   dataset_id: <mdc_dataset_id>
   language: English
   repo_name: default

   training_hp:
     push_to_hub: False
     hub_private_repo: True
     ...
   ```
4. Finetune: `python src/speech_to_text_finetune/finetune_whisper.py`

> [!NOTE]
> To enable downloads via Python API, you must check the acceptance boxes on the MDC platform.

#### Option B: Use a locally downloaded Common Voice dataset

1. Download from https://datacollective.mozillafoundation.org/datasets (newest versions) or https://commonvoice.mozilla.org/en/datasets (older versions), pick your language/version.
2. Extract the zip to a directory on your machine.
3. Set `dataset_id` in `config.yaml` to the local dataset path. Example:
   ```
   model_id: openai/whisper-tiny
   dataset_id: path/to/common_voice_data/language_id
   language: English
   repo_name: default

   training_hp:
     push_to_hub: False
     hub_private_repo: True
     ...
   ```
4. Finetune: `python src/speech_to_text_finetune/finetune_whisper.py`

> [!TIP]
> Run `python demo/model_comparison_app.py` to easily compare the performance of two models side by side ([example](images/model_comparison_example.png)).

## Troubleshooting

If you are having issues / bugs, check our [Troubleshooting](https://mozilla-ai.github.io/speech-to-text-finetune/getting-started/#troubleshooting) section, before opening a new issue.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.

