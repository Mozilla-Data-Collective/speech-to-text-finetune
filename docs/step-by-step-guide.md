# **Step-by-Step Guide: How the Speech-to-Text-Finetune Blueprint Works**

This Blueprint enables you to fine-tune a Speech-to-Text (STT) model, using either your own data or the Common Voice dataset. This Step-by-Step guide walks you through the end-to-end process of finetuning an STT model based on your needs.

---

## **Overview**
This blueprint consists of three independent, yet complementary, components:

1. **Transcription app** 🎙️📝: A simple UI that lets you record your voice, pick any HF STT/ASR model, and get an instant transcription.
2. **Dataset maker app** 📂🎤: A UI app that enables you to easily and quickly create your own Speech-to-Text dataset.
3. **Finetuning script** 🛠️🤖: A script to finetune your own STT model, either using Common Voice data or your own custom data created by the Dataset maker app.

---

## Prerequisites

- Python deps installed: `pip install -e .` and `ffmpeg` installed:
  - \[Ubuntu\]: `sudo apt install ffmpeg`
  - \[Mac\]: `brew install ffmpeg`
- \[Optional\] Hugging Face login if you plan to track your models: `huggingface-cli login`
- MDC access for Common Voice via the Python SDK:
  - Create an account and get an API key from: https://datacollective.mozillafoundation.org/api-reference
- Local `.env` with MDC API key:
  - `cp example_data/.env.example .env`
  - Edit `.env` and set `MDC_API_KEY=<your_api_key>` from https://datacollective.mozillafoundation.org/api-reference

Visit **[Getting Started](getting-started.md)** for initial project setup.

---

## Step-by-Step Guide

### Step 1 - Initial transcription testing

Initially, you can test the quality of the Speech-to-Text models available in HuggingFace by running the Transcription app.

1. Run:
   ```bash
   python demo/transcribe_app.py
   ```
2. Select or add the HF model id of your choice.
3. Record a sample and inspect the transcription. You may find that there are sometimes inaccuracies for your voice/accent/chosen language, indicating the model could benefit form finetuning on additional data.

### Step 2 - Make your Custom Dataset for STT finetuning

1. Create your own, custom dataset by running this command and following the instructions:

    ```bash
    python src/speech_to_text_finetune/make_custom_dataset_app.py
    ```

2. Follow the instruction in the app to create at least 10 audio samples, which will be saved locally.

### Step 3 - Create a finetuned STT model using your custom data

1. Configure `config.yaml` (example):
   ```bash
   model_id: openai/whisper-tiny
   dataset_id: example_data/custom
   language: English
   repo_name: default

   training_hp:
     push_to_hub: False
     hub_private_repo: True
     ...
   ```

Note that if you select `push_to_hub: True` you need to have an HF account and log in locally.

2. Finetune:
   ```bash
   python src/speech_to_text_finetune/finetune_whisper.py
   ```
> [!TIP]
> You can gracefully stop finetuning with CTRL+C; evaluation/upload steps will still run.

### Step 4 - Create a finetuned STT model using Common Voice

Pick one of the following:

- Option A: Mozilla Data Collective Python SDK
  1. Ensure `.env` contains a valid `MDC_API_KEY`.
  2. Find the MDC dataset id for your language (Scripted or Spontaneous).
  3. Configure `config.yaml` with the MDC dataset id:
     ```bash
     model_id: openai/whisper-tiny
     dataset_id: <mdc_dataset_id>
     language: English
     repo_name: default

     training_hp:
       push_to_hub: False
       hub_private_repo: True
       ...
     ```
  4. Finetune:
     ```bash
     python src/speech_to_text_finetune/finetune_whisper.py
     ```

- Option B: Local Common Voice download
  1. Download from https://datacollective.mozillafoundation.org/datasets and extract locally.
  2. Configure `config.yaml` with the local path:
     ```bash
     model_id: openai/whisper-tiny
     dataset_id: path/to/common_voice_data/language_id
     language: English
     repo_name: default

     training_hp:
       push_to_hub: False
       hub_private_repo: True
       ...
     ```
  3. Finetune:
     ```bash
     python src/speech_to_text_finetune/finetune_whisper.py
     ```

> [!NOTE]
> The first time a dataset is used, it is processed and cached locally. Next runs reuse the processed version to save time and compute.

### Step 5 - Evaluate transcription accuracy with your finetuned STT model

1. Start the Transcription app:
   ```bash
   python demo/transcribe_app.py
   ```
2. Select your HF model id (if pushed) or provide a local model path.
3. Record a sample and compare results.

### Step 6 - Compare transcription performance between two models

1. Start the Model Comparison app:
   ```bash
   python demo/model_comparison_app.py
   ```
2. Select a baseline model and a comparison model (e.g., your finetuned model).
3. Record a sample and review both transcriptions side-by-side.

### Step 7 - Evaluate a model on the Fleurs dataset on a specific language

1. Configure the arguments and run:
   ```bash
   python src/speech_to_text_finetune/evaluate_whisper_fleurs.py --model_id openai/whisper-tiny --lang_code sw_ke --language Swahili --eval_batch_size 8 --n_test_samples -1 --fp16 True --update_hf_repo False
   ```

## 🎨 **Customizing the Blueprint**

To better understand how you can tailor this Blueprint to suit your specific needs, please visit the **[Customization Guide](customization.md)**.

## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!

## 📖 **Resources & References**

If you are interested in learning more about this topic, you might find the following resources helpful:
- [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper) (Blog post by HuggingFace which inspired the implementation of the Blueprint!)

- [Automatic Speech Recognition Course from HuggingFace](https://huggingface.co/learn/audio-course/en/chapter5/introduction) (Series of Blog posts)

- [Fine-Tuning ASR Models: Key Definitions, Mechanics, and Use Cases](https://www.gladia.io/blog/fine-tuning-asr-models) (Blog post by Gladia)

- [Active Learning Approach for Fine-Tuning Pre-Trained ASR Model for a low-resourced Language](https://aclanthology.org/2023.icon-1.9.pdf) (Paper)

- [Exploration of Whisper fine-tuning strategies for low-resource ASR](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00349-3) (Paper)

- [Finetuning Pretrained Model with Embedding of Domain and Language Information for ASR of Very Low-Resource Settings](https://www.worldscientific.com/doi/abs/10.1142/S2717554523500248?download=true&journalCode=ijalp) (Paper)
