# 🎨 **Customization Guide**

This Blueprint is designed to be flexible and easily adaptable to your specific needs. This guide will walk you through some key areas you can customize to make the Blueprint your own.

---

## BYOD: Bring Your Own Dataset

> But I already have my own speech-text dateset! I don't want to create a new one from scratch or use Common Voice!
> Does this Blueprint have anything to offer me?

**But of course!**

This guide will walk you through how to use the existing codebase to adapt it to your own unique dataset with minimal effort.

The idea is to load and pre-process your own dataset in the same format as the existing datasets, allowing you to seamlessly integrate with the `finetune_whisper.py` script.

### Step 1: Understand your Dataset

Before creating your custom dataset loading function, it's essential to understand the data format that the `finetune_whisper.py` script expects. Typically, the dataset should have a structure that looks a bit like this:

```python
{
    "train": [
        {
            "audio": "path/to/audio_file.wav",
            "text": "The transcribed text of the audio"
        },
        # More examples...
    ],
    "test": [
        {
            "audio": "path/to/audio_file_2.wav",
            "text": "Another transcribed text"
        },
        # More examples...
    ]
}
```

Notably, there should be a pair of transcribed text and an audio clip (usually in the form of a path to the audio file, either `.mp3` or `.wav`)

### Step 2: Use the built-in tabular ASR dataset loader

If your dataset is a local `.csv`, `.tsv`, or `.parquet` file (or a directory containing one) with the columns `audio_path` and `transcription`, you do **not** need to write a custom loader anymore. The built-in loader in `data_process.py` will:

- keep only the audio path and transcription columns
- ignore extra metadata columns such as `topic`, `speaker_id`, or `paragraph_id`
- use a `split` column if it already exists
- otherwise create a train/test split using `sklearn.model_selection.train_test_split`

As an example, lets consider that you have a directory with a csv file and all the audio clips like this:

```
datasets/
├── my_dataset/
│   ├── dataset.csv
│   └── clips/
│       ├── audio_1.wav
│       ├── audio_2.wav
│       ├── audio_3.wav
│       └── ...
```
and that the .csv file has the following format:

```
csv my_dataset/dataset.csv
audio_path,transcription,topic,speaker_id
clips/example_1.mp3,"This is an example",culture,speaker_1
clips/example_2.mp3,"This is another example",culture,speaker_2
...
clips/example_n.mp3,"This is yet another example",culture,speaker_n
```

Optionally, you can also provide a `split` column with values like `train`, `dev`, `validation`, `test`, or `eval`. If both train and test are already defined, that split will be preserved.

### Step 3: Update your config file

Point `dataset_id` to either the dataset directory or directly to the dataset file. If your dataset does **not** already define a `split` column with both train and test rows, you can control the generated test split with `test_size`.

```
model_id: openai/whisper-tiny
dataset_id: /home/user/datasets/my_dataset
language: English
repo_name: default
n_train_samples: -1
n_test_samples: -1
test_size: 0.2

training_hp:
  push_to_hub: False
  hub_private_repo: True
  ...

```

### Step 4: Fine-Tune the model with your own dataset!

Finally, simply run the finetune_whisper.py script to fine-tune the Whisper model using your custom dataset.

```
python src/speech_to_text_finetune/finetune_whisper.py
```


## 🤝 **Contributing to the Blueprint**

Want to help improve or extend this Blueprint? Check out the **[Future Features & Contributions Guide](future-features-contributions.md)** to see how you can contribute your ideas, code, or feedback to make this Blueprint even better!
