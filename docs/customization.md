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
- expect `audio_path` to already contain an absolute path to the audio file
- preserve a `split` column if it already defines both train and test data
- otherwise create a fresh train/test split using `sklearn.model_selection.train_test_split`

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
/home/user/datasets/my_dataset/clips/example_1.mp3,"This is an example",culture,speaker_1
/home/user/datasets/my_dataset/clips/example_2.mp3,"This is another example",culture,speaker_2
...
/home/user/datasets/my_dataset/clips/example_n.mp3,"This is yet another example",culture,speaker_n
```

Optionally, you can also provide a `split` column with values like `train`, `dev`, `val`, `valid`, `validation`, `test`, `eval`, or `evaluation`.

### How `test_size` is applied

- If your dataset already defines both train and test rows, that split is preserved and `test_size` is ignored.
- `dev`, `val`, `valid`, and `validation` are treated as training data.
- `test`, `eval`, and `evaluation` are treated as test data.
- If the `split` column is missing, empty, or only defines one side of the split, the loader creates a fresh train/test split over the full dataset.
- In that case, `test_size` is passed directly to `sklearn.model_selection.train_test_split`.
- If `test_size` is `null`, scikit-learn's default split size is used.
- For datasets that already come with fixed splits, such as Common Voice or the legacy `train/` + `test/` custom dataset layout, the existing split is preserved and `test_size` is ignored.

### Step 3: Update your config file

Point `dataset_id` to either the dataset directory or directly to the dataset file. `test_size` is only used when the loader needs to create a train/test split. If your dataset already defines both train and test rows, that split is preserved and `test_size` is ignored. If your dataset does not define a complete split, a fresh split is created from all rows using `test_size` or, when `test_size: null`, scikit-learn's default behavior. If you are using an MDC dataset id instead of a local path, you can also set `download_directory` to choose where the raw dataset should be downloaded.

```
model_id: openai/whisper-tiny
dataset_id: /home/user/datasets/my_dataset
language: English
repo_name: default
download_directory: ""  # Only used for MDC dataset ids
n_train_samples: -1
n_test_samples: -1
test_size: 0.2  # Only used when the loader needs to create a test split

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
