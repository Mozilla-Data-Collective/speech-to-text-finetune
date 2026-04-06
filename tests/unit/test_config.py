import pytest
import yaml
from pydantic import ValidationError

from speech_to_text_finetune.config import Config, TrainingConfig, load_config


@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig(
        push_to_hub=False,
        hub_private_repo=False,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=0,
        gradient_checkpointing=False,
        fp16=False,
        eval_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=16,
        save_steps=1,
        logging_steps=1,
        load_best_model_at_end=False,
        save_total_limit=1,
        metric_for_best_model="wer",
        greater_is_better=False,
    )


@pytest.mark.parametrize("test_size", [0.2, 2, None])
def test_config_accepts_valid_test_size(training_config, test_size):
    cfg = Config(
        model_id="openai/whisper-tiny",
        dataset_id="example_data/custom",
        language="English",
        repo_name="default",
        n_train_samples=-1,
        n_test_samples=-1,
        download_directory="",
        test_size=test_size,
        training_hp=training_config,
    )

    assert cfg.test_size == test_size


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, -1])
def test_config_rejects_invalid_test_size(training_config, test_size):
    with pytest.raises(ValidationError):
        Config(
            model_id="openai/whisper-tiny",
            dataset_id="example_data/custom",
            language="English",
            repo_name="default",
            n_train_samples=-1,
            n_test_samples=-1,
            download_directory="",
            test_size=test_size,
            training_hp=training_config,
        )


def test_config_defaults_download_directory_to_empty_string(training_config):
    cfg = Config(
        model_id="openai/whisper-tiny",
        dataset_id="example_data/custom",
        language="English",
        repo_name="default",
        n_train_samples=-1,
        n_test_samples=-1,
        training_hp=training_config,
    )

    assert cfg.download_directory == ""


def test_load_config_reads_download_directory(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_id": "openai/whisper-tiny",
                "dataset_id": "cmf_example_dataset_id",
                "language": "English",
                "repo_name": "default",
                "n_train_samples": -1,
                "n_test_samples": -1,
                "download_directory": "downloads/mdc",
                "test_size": None,
                "training_hp": {
                    "push_to_hub": False,
                    "hub_private_repo": False,
                    "max_steps": 1,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "warmup_steps": 0,
                    "gradient_checkpointing": False,
                    "fp16": False,
                    "eval_strategy": "steps",
                    "per_device_eval_batch_size": 1,
                    "predict_with_generate": True,
                    "generation_max_length": 16,
                    "save_steps": 1,
                    "logging_steps": 1,
                    "load_best_model_at_end": False,
                    "save_total_limit": 1,
                    "metric_for_best_model": "wer",
                    "greater_is_better": False,
                },
            }
        )
    )

    cfg = load_config(str(config_path))

    assert cfg.download_directory == "downloads/mdc"
