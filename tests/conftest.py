import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from transformers import WhisperProcessor, WhisperFeatureExtractor

from speech_to_text_finetune.data_process import load_dataset_from_dataset_id


@pytest.fixture(scope="session", autouse=True)
def chdir_repo_root():
    # Ensure relative paths (e.g., `example_data/custom`) resolve from the repo root
    repo_root = Path(__file__).resolve().parent.parent
    prev_cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@pytest.fixture(scope="session")
def example_config_whisper_path():
    return str(Path(__file__).parent.parent / "tests/e2e/config_whisper.yaml")


@pytest.fixture(scope="session")
def example_config_mms_path():
    return str(Path(__file__).parent.parent / "tests/e2e/config_mms.yaml")


@pytest.fixture(scope="session")
def custom_data_path():
    return str(Path(__file__).parent.parent / "example_data/custom")


@pytest.fixture(scope="session")
def local_common_voice_data_path():
    return str(
        Path(__file__).parent.parent / "example_data/example_cv_dataset/language_id/"
    )


@pytest.fixture(scope="session")
def custom_dataset_half_split(custom_data_path):
    return load_dataset_from_dataset_id(dataset_id=custom_data_path)[0]


def _create_tabular_asr_dataset(
    dataset_dir: Path, source_audio_dir: Path, include_split: bool
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(source_audio_dir.glob("*.wav"))

    dataset_df = pd.DataFrame(
        {
            "audio_path": [str(audio_file.resolve()) for audio_file in audio_files],
            "transcription": [f"sample {idx}" for idx, _ in enumerate(audio_files)],
        }
    )

    if include_split:
        dataset_df["split"] = ["train", "train", "dev", "test", "test"]

    dataset_path = dataset_dir / "dataset.csv"
    dataset_df.to_csv(dataset_path, index=False)
    return dataset_path


@pytest.fixture
def tabular_asr_dataset_path(tmp_path, custom_data_path):
    dataset_dir = tmp_path / "tabular_asr_dataset"
    _create_tabular_asr_dataset(
        dataset_dir=dataset_dir,
        source_audio_dir=Path(custom_data_path) / "train" / "clips",
        include_split=False,
    )
    return str(dataset_dir)


@pytest.fixture
def tabular_asr_dataset_with_split_path(tmp_path, custom_data_path):
    dataset_dir = tmp_path / "tabular_asr_dataset_with_split"
    _create_tabular_asr_dataset(
        dataset_dir=dataset_dir,
        source_audio_dir=Path(custom_data_path) / "train" / "clips",
        include_split=True,
    )
    return str(dataset_dir)


@pytest.fixture
def mock_whisper_processor():
    mock_processor = MagicMock(spec=WhisperProcessor)
    mock_processor.feature_extractor = MagicMock(spec=WhisperFeatureExtractor)
    mock_processor.feature_extractor.sampling_rate = 16000
    mock_processor.side_effect = lambda audio, sampling_rate, text: {
        "input_features": [[0.1] * 80],
        "labels": text,
        "sentence": text,
        "input_length": len(audio) / sampling_rate,
    }
    return mock_processor
