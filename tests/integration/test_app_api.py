# tests/integration/test_app_api.py
import pytest
from unittest.mock import MagicMock

# Import constants defined in app.py if needed, or redefine them
TTS_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_BIT_DEPTH = 16

def test_synthesize_pcm_success(client, mocker):
    """
    Test the /synthesize_pcm endpoint for a successful request
    with mocked audio generation.
    """
    input_text = "Hello world, this is a test."
    expected_voice = "af_heart" # Default or specified
    expected_speed = 1.0       # Default or specified

    # Mock the stream_tts_audio generator function within the app module
    mock_audio_chunk1 = b'\x01\x00\x02\x00\x03\x00' # Example PCM bytes
    mock_audio_chunk2 = b'\x04\x00\x05\x00\x06\x00'
    mock_stream_generator = MagicMock()
    # Configure the mock generator to yield specific byte chunks
    mock_stream_generator.__iter__.return_value = iter([mock_audio_chunk1, mock_audio_chunk2])

    # Patch the function in the 'app' module where it's defined/used
    mock_stream_tts = mocker.patch('app.stream_tts_audio', return_value=mock_stream_generator)

    # Act: Make the POST request
    response = client.post('/synthesize_pcm', json={
        'text': input_text,
        'voice': expected_voice,
        'speed': expected_speed
    })

    # Assert: Check response status and headers
    assert response.status_code == 200
    assert response.mimetype == 'audio/pcm'
    assert response.headers.get('X-Audio-Sample-Rate') == str(TTS_SAMPLE_RATE)
    assert response.headers.get('X-Audio-Channels') == str(AUDIO_CHANNELS)
    assert response.headers.get('X-Audio-Bit-Depth') == str(AUDIO_BIT_DEPTH)
    assert 'Access-Control-Expose-Headers' in response.headers # Check CORS header exposure

    # Assert: Check response data (concatenated chunks from the mock)
    expected_data = mock_audio_chunk1 + mock_audio_chunk2
    assert response.data == expected_data

    # Assert: Check if the mocked function was called correctly
    mock_stream_tts.assert_called_once_with(input_text, expected_voice, expected_speed)

def test_synthesize_pcm_missing_text(client):
    """
    Test the /synthesize_pcm endpoint returns 400 if 'text' is missing.
    """
    response = client.post('/synthesize_pcm', json={
        'voice': 'af_heart',
        'speed': 1.0
    })

    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'Missing \'text\'' in json_data['error']

def test_synthesize_pcm_not_json(client):
    """
    Test the /synthesize_pcm endpoint returns 400 if request is not JSON.
    """
    response = client.post('/synthesize_pcm', data="this is not json")

    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert 'Request must be JSON' in json_data['error']

# TODO: Add tests for invalid speed values, different voices, etc.
# TODO: Add tests for cases where stream_tts_audio yields empty bytes or raises errors (if applicable)
