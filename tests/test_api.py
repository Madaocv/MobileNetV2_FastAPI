import pytest
from fastapi.testclient import TestClient
from main import app
import os
client = TestClient(app)


def test_predict_valid_image():
    response = client.post(
        "/predict?top_k=3",
        files={"file": ("cat.jpeg", open("img/cat.jpeg", "rb"), "image/jpeg")}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "predictions" in json_response
    assert len(json_response["predictions"]) == 3


def test_predict_invalid_file_format():
    with open("img/test.txt", "w") as f:
        f.write("this is a test file and not a valid image")
    response = client.post(
        "/predict?top_k=3",
        files={"file": ("test.txt", open("img/test.txt", "rb"), "text/plain")}
    )
    assert response.status_code == 400
    json_response = response.json()
    assert json_response["detail"] == "Invalid image format. Only jpg and png are supported."
    os.remove("img/test.txt")


def test_predict_corrupted_image():
    with open("img/corrupted_image.jpg", "w") as f:
        f.write("this is not a valid image content")
    response = client.post(
        "/predict?top_k=3",
        files={"file": ("corrupted_image.jpg", open("img/corrupted_image.jpg", "rb"), "image/jpeg")}
    )
    assert response.status_code == 400
    json_response = response.json()
    assert json_response["detail"] == "Could not open image. Make sure the file is not corrupted."
    os.remove("img/corrupted_image.jpg")


def test_predict_top_k_parameter():
    response = client.post(
        "/predict?top_k=5",
        files={"file": ("PCB.png", open("img/PCB.png", "rb"), "image/jpeg")}
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "predictions" in json_response
    assert len(json_response["predictions"]) == 5
