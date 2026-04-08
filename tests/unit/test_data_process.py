import os
from pathlib import Path

import pandas as pd
import pytest
from datasets import DatasetDict, Dataset

import speech_to_text_finetune.data_process as data_process
from speech_to_text_finetune.data_process import (
    load_dataset_from_dataset_id,
    load_subset_of_dataset,
    try_find_processed_version,
    process_dataset_for_whisper,
)


def test_try_find_processed_version_mdc():
    # For an MDC dataset id with no local processed copy, this should return None
    dataset = try_find_processed_version(
        dataset_id="mozilla/cv_dummy_dataset_id", language_id="en"
    )
    assert dataset is None


def _assert_proper_dataset(dataset: DatasetDict) -> None:
    assert isinstance(dataset, DatasetDict)
    assert "sentence" in dataset["train"].features
    assert "audio" in dataset["train"].features

    assert "sentence" in dataset["test"].features
    assert "audio" in dataset["test"].features


def test_load_dataset_from_dataset_id_local_cv(local_common_voice_data_path):
    dataset, _ = load_dataset_from_dataset_id(dataset_id=local_common_voice_data_path)
    _assert_proper_dataset(dataset)


def test_load_dataset_from_dataset_id_custom(custom_data_path):
    dataset, _ = load_dataset_from_dataset_id(dataset_id=custom_data_path)
    _assert_proper_dataset(dataset)


def test_load_dataset_from_dataset_id_tabular_asr_dataset(tabular_asr_dataset_path):
    dataset, _ = load_dataset_from_dataset_id(dataset_id=tabular_asr_dataset_path)

    _assert_proper_dataset(dataset)
    assert len(dataset["train"]) == 3
    assert len(dataset["test"]) == 2
    assert set(dataset["train"].column_names) == {"audio", "sentence"}
    assert set(dataset["test"].column_names) == {"audio", "sentence"}
    assert dataset["train"][0]["audio"].startswith("/")


def test_load_dataset_from_dataset_id_tabular_asr_dataset_with_float_test_size(
    tabular_asr_dataset_path,
):
    dataset, proc_dataset_path = load_dataset_from_dataset_id(
        dataset_id=tabular_asr_dataset_path,
        test_size=0.4,
    )

    _assert_proper_dataset(dataset)
    assert len(dataset["train"]) == 3
    assert len(dataset["test"]) == 2
    assert proc_dataset_path.name == "processed_version_test_size_0_4"


def test_load_dataset_from_dataset_id_tabular_asr_dataset_with_int_test_size(
    tabular_asr_dataset_path,
):
    dataset, proc_dataset_path = load_dataset_from_dataset_id(
        dataset_id=tabular_asr_dataset_path,
        test_size=1,
    )

    _assert_proper_dataset(dataset)
    assert len(dataset["train"]) == 4
    assert len(dataset["test"]) == 1
    assert proc_dataset_path.name == "processed_version_test_size_1"


def test_load_dataset_from_dataset_id_tabular_asr_dataset_with_split(
    tabular_asr_dataset_with_split_path,
):
    dataset, _ = load_dataset_from_dataset_id(
        dataset_id=tabular_asr_dataset_with_split_path
    )

    _assert_proper_dataset(dataset)
    assert len(dataset["train"]) == 3
    assert len(dataset["test"]) == 2
    assert set(dataset["train"].column_names) == {"audio", "sentence"}
    assert set(dataset["test"].column_names) == {"audio", "sentence"}


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipped in GitHub Actions",
)
def test_load_dataset_from_dataset_id_tabular_asr_csv_file(
    tabular_asr_dataset_file_path,
):
    dataset, proc_dataset_path = load_dataset_from_dataset_id(
        dataset_id=tabular_asr_dataset_file_path
    )

    _assert_proper_dataset(dataset)
    assert len(dataset["train"]) == 3
    assert len(dataset["test"]) == 2
    assert proc_dataset_path.name == "processed_version"
    assert proc_dataset_path.parent.name == "tabular_asr_dataset_file"


def test_load_dataset_from_dataset_id_tabular_asr_dataset_rejects_relative_audio_path(
    tmp_path,
):
    dataset_dir = tmp_path / "relative_tabular_asr_dataset"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "audio_path": ["clips/example.wav", "clips/example_2.wav"],
            "transcription": ["sample 1", "sample 2"],
        }
    ).to_csv(dataset_dir / "dataset.csv", index=False)

    with pytest.raises(ValueError, match="absolute audio_path"):
        load_dataset_from_dataset_id(dataset_id=str(dataset_dir))


def test_load_dataset_from_dataset_id_mdc_generic_asr(
    monkeypatch, tmp_path, custom_data_path
):
    source_audio_file = next((Path(custom_data_path) / "train" / "clips").glob("*.wav"))

    dataset_df = pd.DataFrame(
        {
            "audio_path": [
                str(source_audio_file.resolve()),
                str(source_audio_file.resolve()),
            ],
            "transcription": ["mdc train", "mdc test"],
            "split": ["train", "test"],
            "topic": ["culture", "culture"],
        }
    )

    monkeypatch.setenv("MDC_API_KEY", "dummy-key")
    captured_kwargs = {}

    def mock_load_dataset(dataset_id, download_directory=""):
        captured_kwargs["dataset_id"] = dataset_id
        captured_kwargs["download_directory"] = download_directory
        return dataset_df

    monkeypatch.setattr(data_process, "load_dataset", mock_load_dataset)

    dataset, proc_dataset_path = load_dataset_from_dataset_id(
        dataset_id="mozilla/common-voice-like-mdc",
        download_directory="/tmp/mdc-downloads",
    )

    _assert_proper_dataset(dataset)
    assert len(dataset["train"]) == 1
    assert len(dataset["test"]) == 1
    assert dataset["train"][0]["sentence"] == "mdc train"
    assert dataset["test"][0]["sentence"] == "mdc test"
    assert dataset["train"][0]["audio"] == str(source_audio_file.resolve())
    assert captured_kwargs == {
        "dataset_id": "mozilla/common-voice-like-mdc",
        "download_directory": "/tmp/mdc-downloads",
    }
    assert proc_dataset_path == Path(
        "artifacts/mozilla_common-voice-like-mdc/processed_version"
    )


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipped in GitHub Actions",
)
def test_try_find_processed_version_uses_test_size_specific_cache(
    tabular_asr_dataset_path,
    mock_whisper_processor,
):
    dataset, proc_dataset_dir = load_dataset_from_dataset_id(
        dataset_id=tabular_asr_dataset_path,
        test_size=1,
    )
    process_dataset_for_whisper(
        dataset=dataset,
        processor=mock_whisper_processor,
        batch_size=1,
        proc_dataset_path=proc_dataset_dir,
        num_proc=None,
    )

    cached_dataset = try_find_processed_version(
        dataset_id=tabular_asr_dataset_path,
        test_size=1,
    )
    default_cached_dataset = try_find_processed_version(
        dataset_id=tabular_asr_dataset_path
    )

    assert isinstance(cached_dataset, DatasetDict)
    assert default_cached_dataset is None


def test_load_subset_of_dataset_train(custom_dataset_half_split):
    subset = load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=-1)

    assert len(subset) == len(custom_dataset_half_split["train"]) == 5

    subset = load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=5)
    assert len(subset) == len(custom_dataset_half_split["train"]) == 5

    subset = load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=2)
    assert len(subset) == 2

    subset = load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=0)
    assert len(subset) == 0

    subset = load_subset_of_dataset(custom_dataset_half_split["test"], n_samples=-1)
    assert len(subset) == len(custom_dataset_half_split["test"]) == 5

    with pytest.raises(IndexError):
        load_subset_of_dataset(custom_dataset_half_split["train"], n_samples=6)


@pytest.fixture
def mock_dataset():
    data = {
        "audio": [
            {"array": [0.0] * 16000 * 31, "sampling_rate": 16000},  # 31 seconds
            {"array": [0.0] * 16000 * 29, "sampling_rate": 16000},  # 29 seconds
            {"array": [0.0] * 16000 * 29, "sampling_rate": 16000},  # 29 seconds
        ],
        "sentence": [
            "This is an invalid audio sample.",
            "This is a valid audio sample.",
            "This is a really long text. So long that its actually impossible for Whisper to fully generate such a "
            "long text, meaning that this text should be removed from the dataset. Yeap. Exactly. Completely removed."
            "But actually, because we are mocking the processor, and we are just returning as tokenized labels, this"
            "text itself as-is (see how mock_whisper_processor is implemented), its this text itself that needs to be "
            "longer than 448 (the max generation length of whisper) not the tokenized version of it.",
        ],
    }
    return DatasetDict({"train": Dataset.from_dict(data)})


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipped in GitHub Actions",
)
def test_remove_long_audio_and_transcription_samples(
    mock_dataset, mock_whisper_processor, tmp_path
):
    processed_dataset = process_dataset_for_whisper(
        dataset=mock_dataset,
        processor=mock_whisper_processor,
        batch_size=1,
        proc_dataset_path=str(tmp_path),
        num_proc=None,
    )
    assert len(processed_dataset["train"]) == 1
    assert processed_dataset["train"][0]["sentence"] == "This is a valid audio sample."
