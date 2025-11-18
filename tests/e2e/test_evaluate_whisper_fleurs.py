import shutil
from pathlib import Path
from speech_to_text_finetune.evaluate_whisper_fleurs import evaluate_fleurs


def test_evaluate_fleurs_e2e():
    results = evaluate_fleurs(
        model_id="openai/whisper-tiny",
        lang_code="af_za",
        language="Afrikaans",
        eval_batch_size=16,
        n_test_samples=10,
        fp16=False,
    )

    expected_dir_path = Path("artifacts/google_fleurs_af_za/processed_version")
    assert expected_dir_path.exists()

    assert 5.0 < results["eval_loss"] < 6.0
    assert 81.24 < results["eval_wer"] < 82.26
    assert 84.83 < results["eval_wer_ortho"] < 84.85
    assert 34.44 < results["eval_cer"] < 34.46
    assert 36.82 < results["eval_cer_ortho"] < 36.84

    shutil.rmtree("artifacts")
