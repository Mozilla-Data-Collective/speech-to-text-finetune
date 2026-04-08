import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from datacollective import load_dataset
from datasets import load_dataset as load_hf_dataset
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers import WhisperProcessor, Wav2Vec2Processor

from speech_to_text_finetune.config import PROC_DATASET_DIR


def try_find_processed_version(
    dataset_id: str,
    language_id: str | None = None,
    test_size: float | int | None = None,
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

    proc_dataset_path = _get_local_proc_dataset_path(dataset_id, test_size=test_size)
    if Path(proc_dataset_path).is_dir():
        return load_from_disk(proc_dataset_path)

    mdc_proc_dataset_path = _get_mdc_proc_dataset_path(dataset_id, test_size=test_size)
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


def _get_mdc_proc_dataset_path(
    dataset_id: str, test_size: float | int | None = None
) -> Path:
    return Path(
        f"./artifacts/{dataset_id.replace('/', '_')}/{_get_proc_dataset_dir_name(test_size)}"
    )


def _get_hf_proc_dataset_path(dataset_id: str, language_id: str | None) -> str:
    hf_proc_path = f"./artifacts/{dataset_id.replace('/', '_')}"
    if language_id:
        hf_proc_path += f"_{language_id}"
    hf_proc_path += f"/{PROC_DATASET_DIR}"
    return hf_proc_path


def _get_local_proc_dataset_path(
    dataset_id: str, test_size: float | int | None = None
) -> Path:
    dataset_path = Path(dataset_id).resolve()
    if dataset_path.is_file():
        return dataset_path.parent / _get_proc_dataset_dir_name(test_size)
    return dataset_path / _get_proc_dataset_dir_name(test_size)


def _get_proc_dataset_dir_name(test_size: float | int | None = None) -> str:
    if test_size is None:
        return PROC_DATASET_DIR

    normalized_test_size = str(test_size).replace(".", "_")
    return f"{PROC_DATASET_DIR}_test_size_{normalized_test_size}"


def load_dataset_from_dataset_id(
    dataset_id: str,
    test_size: float | int | None = None,
    download_directory: str = "",
) -> Tuple[DatasetDict, Path]:
    """
    This function loads a dataset, based on the dataset_id and the content of its directory (if it is a local path).
    Possible cases:
    1. The dataset_id is an MDC dataset id. In that case, an .env file with MDC_API_KEY must be set up.

    2. The dataset_id is a path to a local, Common Voice dataset directory.

    3. The dataset_id is a path to a local, custom dataset directory.

    Args:
        dataset_id: Path to a processed dataset directory or local dataset directory or MDC dataset ID.
        test_size: Optional test_size to use when loading the dataset.
            Only applicable for MDC and tabular ASR datasets that
            don't already contain a usable train/test split.
            Ignored if the dataset already defines both train and test splits.
            If not provided, sklearn.model_selection.train_test_split uses its default behavior
            when creating a train/test split.
        download_directory: Local directory used by the MDC SDK when downloading
            datasets referenced by MDC dataset IDs.

    Returns:
        DatasetDict: A processed dataset ready for training with train/test splits
        Path: Path to save the processed directory

    Raises:
        ValueError: If the dataset cannot be found locally or on MDC
    """

    try:
        dataset = _load_mdc_dataset(
            dataset_id,
            test_size=test_size,
            download_directory=download_directory,
        )
        return dataset, _get_mdc_proc_dataset_path(dataset_id, test_size=test_size)
    except Exception as e:
        # MDC load failed (dataset not present on MDC or transient MDC error) — try next loaders.
        logger.debug(f"MDC load skipped for {dataset_id}: \n{e}")

    try:
        dataset = _load_local_common_voice(dataset_id)
        return dataset, _get_local_proc_dataset_path(dataset_id, test_size=test_size)
    except FileNotFoundError:
        # Not a local Common Voice dataset — try next loader.
        pass
    except ValueError:
        # Unexpected dataset format — try next loader.
        pass

    try:
        dataset = _load_custom_dataset(dataset_id, test_size=test_size)
        return dataset, _get_local_proc_dataset_path(dataset_id, test_size=test_size)
    except FileNotFoundError:
        pass

    raise ValueError(
        f"There was an error trying to load the dataset: {dataset_id}. "
        f"If the dataset id is a valid MDC identifier, please check that:\n"
        f"- you have agreed to the terms & conditions of the specific dataset.\n"
        f"- you have set your MDC_API_KEY environment variable.\n"
        f"If the dataset id is a local path, please check that you are using the absolute path and it exists."
    )


def _load_mdc_dataset(
    dataset_id: str,
    test_size: float | int | None = None,
    download_directory: str = "",
) -> DatasetDict:
    """
    Load a dataset from MDC and normalize it into the train/test format
    expected by the fine-tuning scripts.

    Valid MDC ASR datasets are expected to be tabular ASR datasets containing
    `audio_path` and `transcription`, and optionally `split`.

    Args:
        dataset_id: dataset id from the Mozilla Data Collective
        download_directory: local directory where the MDC SDK should download the raw dataset

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

    dataset_df = load_dataset(dataset_id, download_directory=download_directory)
    if not _is_valid_asr_dataset(dataset_df):
        raise ValueError(
            "Unsupported MDC dataset format. Expected an ASR dataset with "
            "`audio_path` and `transcription` columns."
        )

    return _build_asr_dataset_from_df(
        dataset_df=dataset_df,
        audio_clip_column="audio_path",
        text_column="transcription",
        test_size=test_size,
    )


def _is_valid_asr_dataset(dataset_df: pd.DataFrame) -> bool:
    return {"audio_path", "transcription"}.issubset(dataset_df.columns)


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

    if "audios" in dir_names and any(
        name.startswith("ss-corpus") and name.endswith(".tsv") for name in file_names
    ):
        return True
    elif "clips" in dir_names and {"train.tsv", "dev.tsv", "test.tsv"}.issubset(
        file_names
    ):
        return False
    else:
        raise ValueError(
            "Unexpected dataset format. Could not determine if local Common Voice is SPS or SCS."
        )


def _load_local_common_voice(cv_data_dir: str) -> DatasetDict:
    """
    Shared loader for local Common Voice (SPS/SCS).
    Build a single DataFrame with `splits` for local Common Voice.
    - SPS: scan `ss-corpus*.tsv` that already contains splits
    - SCS: read `train.tsv`, `dev.tsv`, `test.tsv` and add a `splits` column
    """
    cv_data_dir = Path(cv_data_dir)
    if not cv_data_dir.is_dir():
        raise FileNotFoundError(
            "Local Common Voice datasets must be provided as a directory."
        )

    is_spontaneous_speech = _check_if_local_common_voice_is_spontaneous(cv_data_dir)

    if is_spontaneous_speech:
        dataset_df = None
        for file in cv_data_dir.iterdir():
            if (
                file.is_file()
                and file.name.startswith("ss-corpus")
                and file.name.endswith(".tsv")
            ):
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
        raise ValueError("Expected a 'split' column in the dataset DataFrame.")

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


def _build_asr_dataset_from_df(
    dataset_df: pd.DataFrame,
    audio_clip_column: str = "audio_path",
    text_column: str = "transcription",
    test_size: float | int | None = None,
) -> DatasetDict:
    """
    Build a DatasetDict from a tabular ASR dataset.
    Keeps only the audio path and transcription columns, normalizes them to
    `audio` and `sentence`, and ensures train/test splits exist.
    """
    df = dataset_df.copy()
    df = df.rename(columns={text_column: "sentence"})
    df = _ensure_train_test_split(df, test_size=test_size)
    df = _rename_audio_column(df=df, audio_clip_column=audio_clip_column)

    train_df = df[df["split"] == "train"][["audio", "sentence"]]
    test_df = df[df["split"] == "test"][["audio", "sentence"]]

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )


def _ensure_train_test_split(
    df: pd.DataFrame,
    split_column: str = "split",
    test_size: float | int | None = None,
) -> pd.DataFrame:
    """
    Normalize an optional split column into train/test.
    If no usable split column exists, create one with sklearn's train_test_split.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    fallback_df = df.drop(columns=[split_column], errors="ignore")
    if split_column not in df.columns:
        return _split_train_test(df, split_column=split_column, test_size=test_size)

    normalized_split = df[split_column].fillna("").astype(str).str.strip().str.lower()
    if normalized_split.eq("").all():
        return _split_train_test(
            fallback_df, split_column=split_column, test_size=test_size
        )

    split_map = {
        "train": "train",
        "dev": "train",
        "val": "train",
        "valid": "train",
        "validation": "train",
        "test": "test",
        "eval": "test",
        "evaluation": "test",
    }
    mapped_split = normalized_split.map(split_map)
    invalid_splits = sorted(set(normalized_split[mapped_split.isna()]))
    if invalid_splits:
        raise ValueError(
            "Unsupported split values found in dataset: " + ", ".join(invalid_splits)
        )

    df = df.copy()
    df[split_column] = mapped_split

    if len(df) == 1:
        df[split_column] = "train"
        return df

    if {"train", "test"}.issubset(set(mapped_split)):
        return df

    logger.warning(
        "Dataset split column did not define both train and test splits. "
        "Falling back to sklearn train_test_split."
    )
    return _split_train_test(
        fallback_df, split_column=split_column, test_size=test_size
    )


def _split_train_test(
    df: pd.DataFrame,
    split_column: str = "split",
    test_size: float | int | None = None,
) -> pd.DataFrame:
    if len(df) == 1:
        return df.assign(**{split_column: "train"})

    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    except ValueError as exc:
        raise ValueError(
            f"Could not create a train/test split with test_size={test_size!r}."
        ) from exc

    return pd.concat(
        [
            train_df.assign(**{split_column: "train"}),
            test_df.assign(**{split_column: "test"}),
        ],
        ignore_index=True,
    )


def _rename_audio_column(df: pd.DataFrame, audio_clip_column: str) -> pd.DataFrame:
    """
    Rename the audio column to `audio` for tabular ASR datasets whose
    `audio_path` values are already absolute paths.
    """
    df = df.rename(columns={audio_clip_column: "audio"})
    if df["audio"].isna().any() or df["audio"].astype(str).str.strip().eq("").any():
        raise ValueError("Audio path cannot be empty.")
    if not df["audio"].astype(str).map(os.path.isabs).all():
        raise ValueError(
            "Tabular ASR datasets must provide absolute audio_path values."
        )
    return df


def _join_audio_path(
    audio_dir: Path | List[Path] | Tuple[Path, ...], rel_path: str
) -> str:
    """
    Safely join one or more base audio directories with the relative file path.
    Keeps absolute paths as-is and resolves the result.
    """
    if pd.isna(rel_path) or not str(rel_path).strip():
        raise ValueError("Audio path cannot be empty.")

    p = str(rel_path)
    if os.path.isabs(p):
        return str(Path(p).resolve())

    bases = audio_dir if isinstance(audio_dir, (list, tuple)) else [audio_dir]
    resolved_candidates = [(Path(base) / p).resolve() for base in bases]
    for candidate in resolved_candidates:
        if candidate.exists():
            return str(candidate)

    return str(resolved_candidates[0])


def _replace_rel_path_with_abs_path(
    df: pd.DataFrame,
    audio_dir: str | Path | List[str | Path] | Tuple[str | Path, ...],
    audio_clip_column: str,
) -> pd.DataFrame:
    """
    Rename the audio column and convert relative file paths to absolute paths.
    """
    df = df.rename(columns={audio_clip_column: "audio"})
    if isinstance(audio_dir, (str, Path)):
        base_dirs: List[Path] = [Path(audio_dir)]
    else:
        base_dirs = [Path(base_dir) for base_dir in audio_dir]

    df["audio"] = df["audio"].apply(lambda p: _join_audio_path(base_dirs, p))
    return df


def _get_audio_files_from_dir(dataset_dir: Path) -> List[str]:
    return sorted(
        [
            str(f.resolve())
            for f in dataset_dir.iterdir()
            if f.suffix == ".wav" or f.suffix == ".mp3"
        ],
    )


def _load_custom_dataset(
    dataset_dir: str, test_size: float | int | None = None
) -> DatasetDict:
    """
    Load sentences and accompanied recorded audio files into a pandas DataFrame, then split into train/test and finally
    load it into two distinct train Dataset and test Dataset.

    Sentences and audio files should be indexed like this: <index>: <sentence> should be accompanied by rec_<index>.wav

    Args:
        dataset_dir (str): path to the local dataset, expecting a text.csv and .wav files under the directory

    Returns:
        DatasetDict: HF Dataset dictionary that consists of two distinct Datasets (train+validation and test)
    """
    dataset_path = Path(dataset_dir)

    if _has_legacy_custom_dataset_structure(dataset_path):
        train_file = dataset_path / "train" / "text.csv"
        train_dir = dataset_path / "train" / "clips"
        test_file = dataset_path / "test" / "text.csv"
        test_dir = dataset_path / "test" / "clips"

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        train_df["audio"] = _get_audio_files_from_dir(train_dir)
        test_df["audio"] = _get_audio_files_from_dir(test_dir)

        return DatasetDict(
            {
                "train": Dataset.from_pandas(
                    train_df[["audio", "sentence"]], preserve_index=False
                ),
                "test": Dataset.from_pandas(
                    test_df[["audio", "sentence"]], preserve_index=False
                ),
            }
        )

    tabular_dataset_path = _get_custom_tabular_dataset_path(dataset_path)
    if tabular_dataset_path is None:
        raise FileNotFoundError(
            "Could not find a supported local dataset format. Expected either the "
            "legacy train/test layout or a tabular file with `audio_path` and `transcription` columns."
        )

    dataset_df = _read_tabular_dataset(tabular_dataset_path)

    return _build_asr_dataset_from_df(
        dataset_df=dataset_df,
        audio_clip_column="audio_path",
        text_column="transcription",
        test_size=test_size,
    )


def _has_legacy_custom_dataset_structure(dataset_path: Path) -> bool:
    return dataset_path.is_dir() and all(
        required_path.exists()
        for required_path in [
            dataset_path / "train" / "text.csv",
            dataset_path / "train" / "clips",
            dataset_path / "test" / "text.csv",
            dataset_path / "test" / "clips",
        ]
    )


def _get_custom_tabular_dataset_path(dataset_path: Path) -> Path | None:
    if dataset_path.is_file():
        try:
            dataset_df = _read_tabular_dataset(dataset_path)
        except ValueError:
            return None
        return dataset_path if _is_valid_asr_dataset(dataset_df) else None

    if not dataset_path.is_dir():
        return None

    candidate_paths = sorted(
        file_path
        for file_path in dataset_path.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in {".csv", ".tsv", ".parquet"}
    )
    for candidate_path in candidate_paths:
        dataset_df = _read_tabular_dataset(candidate_path)
        if _is_valid_asr_dataset(dataset_df):
            return candidate_path

    return None


def _read_tabular_dataset(dataset_path: Path) -> pd.DataFrame:
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix == ".tsv":
        return pd.read_csv(dataset_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(dataset_path)

    raise ValueError(
        f"Unsupported dataset file format: {dataset_path.suffix}. "
        "Supported formats are .csv, .tsv and .parquet."
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

    dataset = load_hf_dataset(
        fleurs_dataset_id, language_id, split="test", revision="refs/convert/parquet"
    )
    dataset = load_subset_of_dataset(dataset, n_test_samples)

    dataset = dataset.rename_column(
        original_column_name="raw_transcription", new_column_name="sentence"
    )
    dataset = dataset.select_columns(["audio", "sentence"])

    save_proc_dataset_path = _get_hf_proc_dataset_path(fleurs_dataset_id, language_id)
    logger.info("Processing dataset...")
    dataset = process_dataset_for_whisper(
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


def process_dataset_for_whisper(
    dataset: DatasetDict | Dataset,
    processor: WhisperProcessor,
    batch_size: int,
    proc_dataset_path: str | Path,
    num_proc: int | None = 1,
) -> DatasetDict | Dataset:
    """
    Process dataset to the expected format by a Whisper model and then save it
    locally for future use.
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
        num_proc=num_proc,
    )

    dataset = dataset.filter(
        _is_audio_in_length_range,
        input_columns=["input_length"],
        fn_kwargs={"max_input_length": 30.0},
        num_proc=num_proc,
    )
    dataset = dataset.filter(
        _are_labels_in_length_range,
        input_columns=["labels"],
        fn_kwargs={"max_label_length": 448},
        num_proc=num_proc,
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


def get_mms_dataset_prep_fn(processor):
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # Here, we could add some text cleaning/preprocessing, but currently
        # the assumption is that this will be done prior to fine-tuning.
        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    return prepare_dataset


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
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

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


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
