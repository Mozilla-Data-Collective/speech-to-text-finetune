import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from datacollective import DataCollective
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
from dotenv import load_dotenv
from loguru import logger
from transformers import WhisperProcessor

from speech_to_text_finetune.config import PROC_DATASET_DIR


def try_find_processed_version(
    dataset_id: str, language_id: str | None = None
) -> DatasetDict | Dataset | None:
    """
    Try to load a processed version of the dataset if it exists locally. Check if:
    1. The dataset_id is a local path to a processed dataset directory.
    or
    2. The dataset_id is a path to a local dataset, but a processed version already exists locally.
    or
    3. The dataset_id is an MDC dataset ID, but a processed version already exists locally.
    or
    4. The dataset_id is a HuggingFace dataset ID, but a processed version already exists locally.
    """
    if Path(dataset_id).name == PROC_DATASET_DIR and Path(dataset_id).is_dir():
        if (
            Path(dataset_id + "/train").is_dir()
            and Path(dataset_id + "/test").is_dir()
            and Path(dataset_id + "/dataset_dict.json").is_file()
        ):
            return load_from_disk(dataset_id)
        else:
            raise FileNotFoundError("Processed dataset is incomplete.")

    proc_dataset_path = _get_local_proc_dataset_path(dataset_id)
    if Path(proc_dataset_path).is_dir():
        return load_from_disk(proc_dataset_path)

    mdc_proc_dataset_path = _get_mdc_proc_dataset_path(dataset_id)
    if Path(mdc_proc_dataset_path).is_dir():
        logger.info(
            f"Found processed dataset version at {mdc_proc_dataset_path} of MDC dataset {dataset_id}. "
            f"Loading it directly and skipping processing again the original version."
        )
        return load_from_disk(mdc_proc_dataset_path)

    hf_proc_dataset_path = _get_hf_proc_dataset_path(dataset_id, language_id)
    if Path(hf_proc_dataset_path).is_dir():
        logger.info(
            f"Found processed dataset version at {hf_proc_dataset_path} of HF dataset {dataset_id}. "
            f"Loading it directly and skipping processing again the original version."
        )
        return load_from_disk(hf_proc_dataset_path)

    return None


def _get_mdc_proc_dataset_path(dataset_id: str) -> Path:
    return Path(f"./artifacts/{dataset_id.replace('/', '_')}/{PROC_DATASET_DIR}")


def _get_hf_proc_dataset_path(dataset_id: str, language_id: str | None) -> str:
    hf_proc_path = f"./artifacts/{dataset_id.replace('/', '_')}"
    if language_id:
        hf_proc_path += f"_{language_id}"
    hf_proc_path += f"/{PROC_DATASET_DIR}"
    return hf_proc_path


def _get_local_proc_dataset_path(dataset_id: str) -> Path:
    return Path(dataset_id).resolve() / PROC_DATASET_DIR


def load_dataset_from_dataset_id(
    dataset_id: str
) -> Tuple[DatasetDict, Path]:
    """
    This function loads a dataset, based on the dataset_id and the content of its directory (if it is a local path).
    Possible cases:
    1. The dataset_id is an MDC dataset id. In that case, an .env file with MDC_API_KEY must be set up.

    2. The dataset_id is a path to a local, Common Voice dataset directory.

    3. The dataset_id is a path to a local, custom dataset directory.

    Args:
        dataset_id: Path to a processed dataset directory or local dataset directory or MDC dataset ID.

    Returns:
        DatasetDict: A processed dataset ready for training with train/test splits
        Path: Path to save the processed directory

    Raises:
        ValueError: If the dataset cannot be found locally or on MDC
    """

    try:
        dataset = _load_mdc_common_voice(dataset_id)
        return dataset, _get_mdc_proc_dataset_path(dataset_id)
    except InvalidCommonVoiceDatasetError:
        # This means the dataset was found on MDC but isn't a Common Voice dataset;
        raise
    except Exception as e:
        # MDC load failed (dataset not present on MDC or transient MDC error) — try next loaders.
        logger.debug(f"MDC load skipped for {dataset_id}: \n{e}")

    try:
        dataset = _load_local_common_voice(dataset_id)
        return dataset, _get_local_proc_dataset_path(dataset_id)
    except FileNotFoundError:
        # Not a local Common Voice dataset — try next loader.
        pass

    try:
        dataset = _load_custom_dataset(dataset_id)
        return dataset, _get_local_proc_dataset_path(dataset_id)
    except FileNotFoundError:
        pass

    raise ValueError(
        f"Could not find dataset {dataset_id} locally or at MDC. "
        f"Or you are missing your MDC_API_KEY environment variable."
    )

def _load_mdc_common_voice(dataset_id: str) -> DatasetDict:
    """
    Shared loader for MDC-hosted Common Voice (SPS/SCS).
    Load MDC dataset once and return a single DataFrame with `splits`,
    the audio base directory and the audio column name.

    Args:
        dataset_id: official Common Voice dataset id from the Mozilla Data Collective

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets
        with columns "audio" and "sentence"
    """
    load_dotenv(override=True)
    if "MDC_API_KEY" not in os.environ:
        raise EnvironmentError(
            "MDC_API_KEY environment variable not set. "
            "Please set it to access Mozilla Data Collective datasets."
        )
    mdc_client = DataCollective()
    dataset = mdc_client.load_dataset(dataset_id)
    dataset_df = dataset.to_pandas()
    dataset_details = mdc_client.get_dataset_details(dataset_id)
    data_dir = Path(dataset.corpus_filepath)

    if "spontaneous" in dataset_details["name"].lower():
        is_spontaneous_speech = True
    elif "scripted" in dataset_details["name"].lower():
        is_spontaneous_speech = False
    else:
        raise InvalidCommonVoiceDatasetError(
            "Could not determine if MDC Common Voice dataset is SPS or SCS. "
            "Dataset does not seem to be part of Common Voice collection."
        )

    if is_spontaneous_speech:
        audio_dir = data_dir / "audios"
        audio_clip_column = "audio_file"
    else:
        audio_dir = data_dir / "clips"
        audio_clip_column = "path"

    return _build_cv_dataset_from_df(
        dataset_df=dataset_df,
        audio_dir=audio_dir,
        audio_clip_column=audio_clip_column,
        is_spontaneous_speech=is_spontaneous_speech,
    )

def _check_if_local_common_voice_is_spontaneous(cv_data_dir: Path) -> bool:
    """
    Check if the local Common Voice dataset is Spontaneous (SPS) or Scripted (SCS),
    based on the expected directory structure and file names.
    - SPS: contains `audios/` directory and `ss-corpus*.tsv` files
    - SCS: contains `clips/` directory and `train.tsv`, `dev.tsv`, `test.tsv` files
    """
    entries = list(cv_data_dir.iterdir())
    dir_names = {p.name for p in entries if p.is_dir()}
    file_names = {p.name for p in entries if p.is_file()}

    if "audios" in dir_names and any(name.startswith("ss-corpus") and name.endswith(".tsv") for name in file_names):
        return True
    elif "clips" in dir_names and {"train.tsv", "dev.tsv", "test.tsv"}.issubset(file_names):
        return False
    else:
        raise ValueError("Unexpected dataset format. Could not determine if local Common Voice is SPS or SCS.")

def _load_local_common_voice(cv_data_dir: str) -> DatasetDict:
    """
    Shared loader for local Common Voice (SPS/SCS).
    Build a single DataFrame with `splits` for local Common Voice.
    - SPS: scan `ss-corpus*.tsv` that already contains splits
    - SCS: read `train.tsv`, `dev.tsv`, `test.tsv` and add a `splits` column
    """
    cv_data_dir = Path(cv_data_dir)

    is_spontaneous_speech = _check_if_local_common_voice_is_spontaneous(cv_data_dir)

    if is_spontaneous_speech:
        dataset_df = None
        for file in cv_data_dir.iterdir():
            if file.is_file() and file.name.startswith("ss-corpus") and file.name.endswith(".tsv"):
                dataset_df = pd.read_csv(file, sep="\t")
                break
        if dataset_df is None:
            raise FileNotFoundError("Could not find SPS `ss-corpus*.tsv` file.")
        audio_dir = cv_data_dir / "audios"
        audio_clip_column = "audio_file"
    else:
        train_df = pd.read_csv(cv_data_dir / "train.tsv", sep="\t")
        train_df["split"] = "train"

        dev_df = pd.read_csv(cv_data_dir / "dev.tsv", sep="\t")
        dev_df["split"] = "dev"

        test_df = pd.read_csv(cv_data_dir / "test.tsv", sep="\t")
        test_df["split"] = "test"

        dataset_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
        audio_dir = cv_data_dir / "clips"
        audio_clip_column = "path"

    return _build_cv_dataset_from_df(
        dataset_df=dataset_df,
        audio_dir=audio_dir,
        audio_clip_column=audio_clip_column,
        is_spontaneous_speech=is_spontaneous_speech,
    )


def _build_cv_dataset_from_df(
    dataset_df: pd.DataFrame,
    audio_dir: str | Path,
    audio_clip_column: str,
    is_spontaneous_speech: bool,
) -> DatasetDict:
    """
    Common builder for Common Voice Spontaneous and Scripted Speech, MDC or local.
    - Renames text column to `sentence` (handles SPS's column name `transcription`)
    - Converts audio paths to absolute
    - Merges train+dev and keeps test
    - Returns DatasetDict with only `audio` and `sentence`
    """
    df = dataset_df.copy()

    # Normalize text column name
    if is_spontaneous_speech and "transcription" in df.columns:
        df = df.rename(columns={"transcription": "sentence"})

    # Ensure we have splits
    if "split" not in df.columns:
        raise ValueError("Expected a 'splits' column in the dataset DataFrame.")

    # Convert relative to absolute audio paths
    df = _replace_rel_path_with_abs_path(
        df=df, audio_dir=audio_dir, audio_clip_column=audio_clip_column
    )

    # Split and keep only relevant columns
    train_df = df[df["split"].isin(["train", "dev"])][["audio", "sentence"]]
    test_df = df[df["split"] == "test"][["audio", "sentence"]]

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )

def _join_audio_path(audio_dir: Path, rel_path: str) -> str:
    """
    Safely join base audio directory with the relative file path.
    Keeps absolute paths as-is and resolves the result.
    """
    p = str(rel_path)
    if os.path.isabs(p):
        return p
    return str((audio_dir / p).resolve())

def _replace_rel_path_with_abs_path(
    df: pd.DataFrame, audio_dir: str | Path, audio_clip_column: str
) -> pd.DataFrame:
    """
    Rename the audio column and convert relative file paths to absolute paths.
    """
    df = df.rename(columns={audio_clip_column: "audio"})
    base = Path(audio_dir)
    df["audio"] = df["audio"].apply(lambda p: _join_audio_path(base, p))
    return df


def _get_audio_files_from_dir(dataset_dir: Path) -> List[str]:
    return sorted(
        [
            str(f.resolve())
            for f in dataset_dir.iterdir()
            if f.suffix == ".wav" or f.suffix == ".mp3"
        ],
    )


def _load_custom_dataset(dataset_dir: str) -> DatasetDict:
    """
    Load sentences and accompanied recorded audio files into a pandas DataFrame, then split into train/test and finally
    load it into two distinct train Dataset and test Dataset.

    Sentences and audio files should be indexed like this: <index>: <sentence> should be accompanied by rec_<index>.wav

    Args:
        dataset_dir (str): path to the local dataset, expecting a text.csv and .wav files under the directory

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets (train+validation and test)
    """
    train_file = dataset_dir + "/train/text.csv"
    train_dir = Path(dataset_dir + "/train/clips")
    test_file = dataset_dir + "/test/text.csv"
    test_dir = Path(dataset_dir + "/test/clips")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df["audio"] = _get_audio_files_from_dir(train_dir)
    test_df["audio"] = _get_audio_files_from_dir(test_dir)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def load_and_proc_hf_fleurs(
    language_id: str,
    n_test_samples: int,
    processor: WhisperProcessor,
    eval_batch_size: int,
) -> Dataset:
    """
    Load only the test split of fleurs on a specific language and process it for Whisper.
    Args:
        language_id (str): a registered language identifier from Fleurs
            (see https://huggingface.co/datasets/google/fleurs/blob/main/fleurs.py)
        n_test_samples (int): number of samples to use from the test split
        processor (WhisperProcessor): Processor from Whisper to process the dataset
        eval_batch_size (int): batch size to use for processing the dataset

    Returns:
        DatasetDict: HF Dataset
    """
    fleurs_dataset_id = "google/fleurs"
    if proc_dataset := try_find_processed_version(fleurs_dataset_id, language_id):
        return proc_dataset

    dataset = load_dataset(
        fleurs_dataset_id, language_id, trust_remote_code=True, split="test"
    )
    dataset = load_subset_of_dataset(dataset, n_test_samples)

    dataset = dataset.rename_column(
        original_column_name="raw_transcription", new_column_name="sentence"
    )
    dataset = dataset.select_columns(["audio", "sentence"])

    save_proc_dataset_path = _get_hf_proc_dataset_path(
        fleurs_dataset_id, language_id)
    logger.info("Processing dataset...")
    dataset = process_dataset(
        dataset=dataset,
        processor=processor,
        batch_size=eval_batch_size,
        proc_dataset_path=save_proc_dataset_path,
    )
    logger.info(
        f"Processed dataset saved at {save_proc_dataset_path}. Future runs of {fleurs_dataset_id} will "
        f"automatically use this processed version."
    )
    return dataset


def load_subset_of_dataset(dataset: Dataset, n_samples: int) -> Dataset:
    return dataset.select(range(n_samples)) if n_samples != -1 else dataset


def _is_audio_in_length_range(length: float, max_input_length: float = 30.0) -> bool:
    return 0 < length < max_input_length


def _are_labels_in_length_range(labels: List[int], max_label_length: int = 448) -> bool:
    return len(labels) < max_label_length


def process_dataset(
    dataset: DatasetDict | Dataset,
    processor: WhisperProcessor,
    batch_size: int,
    proc_dataset_path: str | Path,
) -> DatasetDict | Dataset:
    """
    Process dataset to the expected format by a Whisper model and then save it locally for future use.
    """
    # Create a new column that consists of the resampled audio samples in the right sample rate for whisper
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate)
    )

    dataset = dataset.map(
        _process_inputs_and_labels_for_whisper,
        fn_kwargs={"processor": processor},
        remove_columns=dataset.column_names["train"]
        if "train" in dataset.column_names
        else None,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
    )

    dataset = dataset.filter(
        _is_audio_in_length_range,
        input_columns=["input_length"],
        fn_kwargs={"max_input_length": 30.0},
        num_proc=1,
    )
    dataset = dataset.filter(
        _are_labels_in_length_range,
        input_columns=["labels"],
        fn_kwargs={"max_label_length": 448},
        num_proc=1,
    )
    proc_dataset_path = Path(proc_dataset_path)
    Path.mkdir(proc_dataset_path, parents=True, exist_ok=True)
    dataset.save_to_disk(proc_dataset_path)
    return dataset


def _process_inputs_and_labels_for_whisper(
    batch: Dict, processor: WhisperProcessor
) -> Dict:
    """
    Use Whisper's feature extractor to transform the input audio arrays into log-Mel spectrograms
     and the tokenizer to transform the text-label into tokens. This function is expected to be called using
     the .map method in order to process the data batch by batch.
    """
    batched_audio = batch["audio"]

    batch = processor(
        audio=[audio["array"] for audio in batched_audio],
        sampling_rate=processor.feature_extractor.sampling_rate,
        text=batch["sentence"],
    )

    batch["input_length"] = [
        len(audio["array"]) / audio["sampling_rate"] for audio in batched_audio
    ]

    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data Collator class in the format expected by Seq2SeqTrainer used for processing
    input data and labels in batches while finetuning. More info here:
    """

    processor: WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class InvalidCommonVoiceDatasetError(ValueError):
    """Raised when an MDC Common Voice dataset cannot be classified as SPS or SCS."""
    pass
