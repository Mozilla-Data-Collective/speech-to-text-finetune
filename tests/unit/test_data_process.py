from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from datasets import DatasetDict, Dataset

from speech_to_text_finetune.data_process import (
    load_dataset_from_dataset_id,
    load_subset_of_dataset,
    try_find_processed_version,
    process_dataset,
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


def test_load_dataset_from_dataset_id_mdc_cv_sps(tmp_path: Path):
    # Mock MDC CV Spontaneous Speech (SPS): uses the expected dataset structure of
    # 'audios' dir and columns 'audio_file' + 'transcription'
    with patch("speech_to_text_finetune.data_process.DataCollective") as MockDC:
        mock_client = MockDC.return_value

        df = pd.DataFrame(
            {
                "splits": ["train", "dev", "test"],
                "audio_file": ["a.wav", "b.wav", "c.wav"],
                "transcription": ["t1", "t2", "t3"],
            }
        )
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = df

        sps_dir = tmp_path / "sps_dataset"
        (sps_dir / "audios").mkdir(parents=True, exist_ok=True)
        mock_dataset.directory = str(sps_dir)

        mock_client.load_dataset.return_value = mock_dataset
        mock_client.get_dataset_details.return_value = {
            "title": "Common Voice Spontaneous Speech (SPS)"
        }

        dataset, _ = load_dataset_from_dataset_id(dataset_id="mozilla/cv_sps_en")
        _assert_proper_dataset(dataset)


def test_load_dataset_from_dataset_id_mdc_scs(tmp_path: Path):
    # Mock MDC Scripted Speech (SCS): uses the expected dataset structure of
    # 'clips' dir and columns 'path' + 'sentence'
    with patch("speech_to_text_finetune.data_process.DataCollective") as MockDC:
        mock_client = MockDC.return_value

        df = pd.DataFrame(
            {
                "splits": ["train", "dev", "test"],
                "path": ["x.wav", "y.wav", "z.wav"],
                "sentence": ["s1", "s2", "s3"],
            }
        )
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = df

        scs_dir = tmp_path / "scs_dataset"
        (scs_dir / "clips").mkdir(parents=True, exist_ok=True)
        mock_dataset.directory = str(scs_dir)

        mock_client.load_dataset.return_value = mock_dataset
        mock_client.get_dataset_details.return_value = {
            "title": "Common Voice Scripted Speech (SCS)"
        }

        dataset, _ = load_dataset_from_dataset_id(dataset_id="mozilla/cv_scs_en")
        _assert_proper_dataset(dataset)


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


def test_remove_long_audio_and_transcription_samples(
    mock_dataset, mock_whisper_processor, tmp_path
):
    processed_dataset = process_dataset(
        dataset=mock_dataset,
        processor=mock_whisper_processor,
        batch_size=1,
        proc_dataset_path=str(tmp_path),
    )
    assert len(processed_dataset["train"]) == 1
    assert processed_dataset["train"][0]["sentence"] == "This is a valid audio sample."
