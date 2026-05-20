from app.predict import predict_batch_events, predict_single_event


SAMPLE_PAYLOAD = {
    "DER_mass_MMC": 138.4,
    "DER_mass_transverse_met_lep": 51.6,
    "DER_mass_vis": 97.8,
    "PRI_tau_pt": 32.6,
    "PRI_lep_pt": 44.1,
}


def test_predict_single_event_returns_expected_keys():
    result = predict_single_event(payload=SAMPLE_PAYLOAD)

    assert "prediction" in result
    assert "probability" in result
    assert "model_name" in result
    assert "model_version" in result


def test_predict_single_event_returns_valid_prediction_label():
    result = predict_single_event(payload=SAMPLE_PAYLOAD)

    assert result["prediction"] in ["signal", "background"]


def test_predict_single_event_returns_probability():
    result = predict_single_event(payload=SAMPLE_PAYLOAD)

    assert result["probability"] is not None
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert "needs_review" in result
    assert "risk_level" in result


def test_predict_batch_events_returns_predictions_list():
    result = predict_batch_events(payloads=[SAMPLE_PAYLOAD, SAMPLE_PAYLOAD])

    assert "predictions" in result
    assert len(result["predictions"]) == 2

    for prediction in result["predictions"]:
        assert "prediction" in prediction
        assert "probability" in prediction
        assert "model_name" in prediction
        assert "model_version" in prediction
        assert prediction["prediction"] in ["signal", "background"]
        assert isinstance(prediction["probability"], float)
        assert 0.0 <= prediction["probability"] <= 1.0
        assert "needs_review" in prediction
        assert "risk_level" in prediction