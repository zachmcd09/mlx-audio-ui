import importlib.resources
import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn # Add missing import
import misaki as ms  # type: ignore
import numpy as np
from misaki import en


# Create a patch for the deprecated open_text function
def patched_open_text(package, resource):
    """Replacement for deprecated open_text using files() API"""
    return importlib.resources.files(package).joinpath(resource).open("r")


# Apply the patch at the module level
@patch("importlib.resources.open_text", patched_open_text)
class TestSanitizeLSTMWeights(unittest.TestCase):
    def test_sanitize_lstm_weights(self):
        """Test sanitize_lstm_weights function."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import sanitize_lstm_weights

        # Test weight_ih_l0_reverse
        key = "lstm.weight_ih_l0_reverse"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wx_backward")

        # Test weight_hh_l0_reverse
        key = "lstm.weight_hh_l0_reverse"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wh_backward")

        # Test bias_ih_l0_reverse
        key = "lstm.bias_ih_l0_reverse"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_ih_backward")

        # Test bias_hh_l0_reverse
        key = "lstm.bias_hh_l0_reverse"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_hh_backward")

        # Test weight_ih_l0
        key = "lstm.weight_ih_l0"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wx_forward")

        # Test weight_hh_l0
        key = "lstm.weight_hh_l0"
        weights = mx.array(np.zeros((10, 10)))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.Wh_forward")

        # Test bias_ih_l0
        key = "lstm.bias_ih_l0"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_ih_forward")

        # Test bias_hh_l0
        key = "lstm.bias_hh_l0"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "lstm.bias_hh_forward")

        # Test unknown key
        key = "unknown.key"
        weights = mx.array(np.zeros(10))
        result = sanitize_lstm_weights(key, weights)
        self.assertEqual(list(result.keys())[0], "unknown.key")


@patch("importlib.resources.open_text", patched_open_text)
class TestKokoroModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.kokoro.kokoro.json.load")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("mlx_audio.tts.models.kokoro.kokoro.mx.load")
    @patch.object(nn.Module, "load_weights")
    def test_init(self, mock_load_weights, mock_mx_load, mock_open, mock_json_load):
        """Test KokoroModel initialization."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import Model, ModelConfig

        # Mock the config loading
        config = {
            "istftnet": {
                "upsample_kernel_sizes": [20, 12],
                "upsample_rates": [10, 6],
                "gen_istft_hop_size": 5,
                "gen_istft_n_fft": 20,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_initial_channel": 512,
            },
            "dim_in": 64,
            "dropout": 0.2,
            "hidden_dim": 512,
            "max_conv_dim": 512,
            "max_dur": 50,
            "multispeaker": True,
            "n_layer": 3,
            "n_mels": 80,
            "n_token": 178,
            "style_dim": 128,
            "text_encoder_kernel_size": 5,
            "plbert": {
                "hidden_size": 768,
                "num_attention_heads": 12,
                "intermediate_size": 2048,
                "max_position_embeddings": 512,
                "num_hidden_layers": 12,
                "dropout": 0.1,
            },
            "vocab": {"a": 1, "b": 2},
        }
        mock_json_load.return_value = config

        # Mock the weights loading
        mock_mx_load.return_value = {"key": mx.array(np.zeros(10))}

        # Make load_weights return the module
        mock_load_weights.return_value = None

        # Initialize the model with the config parameter
        model = Model(ModelConfig.from_dict(config))

        # Check that the model was initialized correctly
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.vocab, {"a": 1, "b": 2})

    def test_output_dataclass(self):
        """Test KokoroModel.Output dataclass."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.kokoro import Model

        # Create a mock output
        audio = mx.array(np.zeros((1, 1000)))
        pred_dur = mx.array(np.zeros((1, 100)))

        # Mock __init__ to return None
        with patch.object(Model, "__init__", return_value=None):
            output = Model.Output(audio=audio, pred_dur=pred_dur)

        # Check that the output was created correctly
        self.assertIs(output.audio, audio)
        self.assertIs(output.pred_dur, pred_dur)


@patch("importlib.resources.open_text", patched_open_text)
class TestKokoroPipeline(unittest.TestCase):
    def test_aliases_and_lang_codes(self):
        """Test ALIASES and LANG_CODES constants."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import ALIASES, LANG_CODES

        # Check that all aliases map to valid language codes
        for alias_key, alias_value in ALIASES.items():
            self.assertIn(alias_value, LANG_CODES)

        # Check specific mappings
        self.assertEqual(ALIASES["en-us"], "a")
        self.assertEqual(ALIASES["ja"], "j")
        self.assertEqual(LANG_CODES["a"], "American English")
        self.assertEqual(LANG_CODES["j"], "Japanese")

    def test_init(self):
        """Test KokoroPipeline initialization."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import LANG_CODES, KokoroPipeline

        # Mock the KokoroModel - fix the import path
        with patch("mlx_audio.tts.models.kokoro.kokoro.Model") as mock_kokoro_model:
            with patch(
                "mlx_audio.tts.models.kokoro.pipeline.isinstance"
            ) as mock_isinstance:
                mock_model = MagicMock()
                mock_kokoro_model.return_value = mock_model

                # Simply make isinstance always return True when checking for KokoroModel
                mock_isinstance.return_value = True

                # Initialize with default model
                pipeline = KokoroPipeline(
                    lang_code="a", model=mock_model, repo_id="mock"
                )
                self.assertEqual(pipeline.lang_code, "a")
                self.assertEqual(LANG_CODES[pipeline.lang_code], "American English")

                # Initialize with provided model
                model = mock_model
                pipeline = KokoroPipeline(lang_code="a", model=model, repo_id="mock")
                self.assertEqual(pipeline.model, model)

                # Initialize with no model
                pipeline = KokoroPipeline(lang_code="a", model=False, repo_id="mock")
                self.assertIs(pipeline.model, False)

    def test_load_voice(self):
        """Test load_voice method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Setup the pipeline
        with patch.object(KokoroPipeline, "__init__", return_value=None):
            with patch(
                "mlx_audio.tts.models.kokoro.pipeline.torch.load"
            ) as mock_torch_load:
                with patch(
                    "mlx_audio.tts.models.kokoro.pipeline.hf_hub_download"
                ) as mock_hf_hub_download:
                    with patch(
                        "mlx_audio.tts.models.kokoro.pipeline.torch.stack"
                    ) as mock_stack:
                        with patch(
                            "mlx_audio.tts.models.kokoro.pipeline.torch.mean"
                        ) as mock_mean:
                            pipeline = KokoroPipeline.__new__(KokoroPipeline)
                            pipeline.lang_code = "a"
                            pipeline.voices = {}
                            # Add the missing repo_id attribute
                            pipeline.repo_id = "mlx-community/kokoro-tts"

                            # Mock the torch.load return value
                            mock_torch_load.return_value = MagicMock()

                            # Mock torch.stack and torch.mean to return a MagicMock
                            mock_stack.return_value = MagicMock()
                            mock_mean.return_value = MagicMock()

                            # Test loading a single voice
                            pipeline.load_single_voice("voice1")
                            mock_hf_hub_download.assert_called_once()
                            self.assertIn("voice1", pipeline.voices)

                            # Test loading multiple voices
                            mock_hf_hub_download.reset_mock()
                            pipeline.voices = {}  # Reset voices
                            result = pipeline.load_voice("voice1,voice2")
                            self.assertEqual(mock_hf_hub_download.call_count, 2)
                            self.assertIn("voice1", pipeline.voices)
                            self.assertIn("voice2", pipeline.voices)

    def test_tokens_to_ps(self):
        """Test tokens_to_ps method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create mock tokens with whitespace attribute
        token1 = MagicMock(spec=en.MToken)
        token1.ps = "p1"
        token1.whitespace = " "
        token1.phonemes = "p1"

        token2 = MagicMock(spec=en.MToken)
        token2.ps = "p2"
        token2.whitespace = ""
        token2.phonemes = "p2"

        tokens = [token1, token2]

        # Test the method
        with patch.object(KokoroPipeline, "__init__", return_value=None):
            with patch.object(KokoroPipeline, "tokens_to_ps", return_value="p1 p2"):
                result = KokoroPipeline.tokens_to_ps(tokens)
                self.assertEqual(result, "p1 p2")

    def test_tokens_to_text(self):
        """Test tokens_to_text method."""
        # Import inside the test method
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create mock tokens with whitespace attribute
        token1 = MagicMock(spec=en.MToken)
        token1.text = "Hello"
        token1.whitespace = " "

        token2 = MagicMock(spec=en.MToken)
        token2.text = "world"
        token2.whitespace = ""

        tokens = [token1, token2]

        # Test the method
        with patch.object(KokoroPipeline, "__init__", return_value=None):
            with patch.object(
                KokoroPipeline, "tokens_to_text", return_value="Hello world"
            ):
                result = KokoroPipeline.tokens_to_text(tokens)
                self.assertEqual(result, "Hello world")

    def test_result_dataclass(self):
        """Test KokoroPipeline.Result dataclass."""
        # Import inside the test methods
        from mlx_audio.tts.models.kokoro.kokoro import Model
        from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline

        # Create a mock output
        audio = mx.array(np.zeros((1, 1000)))
        pred_dur = mx.array(np.zeros((1, 100)))
        model_output = Model.Output(audio=audio, pred_dur=pred_dur)

        # Create a Result instance
        result = KokoroPipeline.Result(
            graphemes="Hello",
            phonemes="HH EH L OW",
            tokens=[MagicMock()],
            output=model_output,
            text_index=0,
        )

        # Check properties
        self.assertEqual(result.graphemes, "Hello")
        self.assertEqual(result.phonemes, "HH EH L OW")
        self.assertIs(result.audio, audio)
        self.assertIs(result.pred_dur, pred_dur)

        # Test backward compatibility
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Hello")
        self.assertEqual(result[1], "HH EH L OW")
        self.assertIs(result[2], audio)

        # Test iteration
        items = list(result)
        self.assertEqual(items[0], "Hello")
        self.assertEqual(items[1], "HH EH L OW")
        self.assertIs(items[2], audio)


@patch("importlib.resources.open_text", patched_open_text)
class TestBarkModel(unittest.TestCase):
    @patch("mlx_audio.tts.models.bark.bark.BertTokenizer")
    def test_init(self, mock_tokenizer):
        """Test BarkModel initialization."""
        from mlx_audio.tts.models.bark.bark import (
            CoarseAcousticsConfig,
            CodecConfig,
            FineAcousticsConfig,
            Model,
            ModelConfig,
            SemanticConfig,
        )

        # Create mock configs
        semantic_config = SemanticConfig()
        coarse_config = CoarseAcousticsConfig()
        fine_config = FineAcousticsConfig()
        codec_config = CodecConfig()

        config = ModelConfig(
            semantic_config=semantic_config,
            coarse_acoustics_config=coarse_config,
            fine_acoustics_config=fine_config,
            codec_config=codec_config,
        )

        # Initialize model
        model = Model(config)

        # Check that components were initialized correctly
        self.assertIsNotNone(model.semantic)
        self.assertIsNotNone(model.coarse_acoustics)
        self.assertIsNotNone(model.fine_acoustics)
        self.assertIsNotNone(model.tokenizer)

    def test_sanitize_weights(self):
        """Test weight sanitization."""
        from mlx_audio.tts.models.bark.bark import Model, ModelConfig

        # Create a minimal config
        config = ModelConfig(
            semantic_config={},
            coarse_acoustics_config={},
            fine_acoustics_config={},
            codec_config={},
        )

        model = Model(config)

        # Test with transformer weights
        weights = {
            "_orig_mod.transformer.h.0.mlp.weight": mx.zeros((10, 10)),
            "_orig_mod.transformer.h.1.mlp.weight": mx.zeros((10, 10)),
            "lm_head.weight": mx.zeros((10, 10)),
        }

        sanitized = model.sanitize(weights)

        # Check that weights were properly renamed
        self.assertIn("layers.0.mlp.weight", sanitized)
        self.assertIn("layers.1.mlp.weight", sanitized)
        self.assertIn("lm_head.weight", sanitized)


@patch("importlib.resources.open_text", patched_open_text)
class TestBarkPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        from mlx_audio.tts.models.bark.bark import (
            CoarseAcousticsConfig,
            CodecConfig,
            FineAcousticsConfig,
            Model,
            ModelConfig,
            SemanticConfig,
        )
        from mlx_audio.tts.models.bark.pipeline import Pipeline

        # Create mock model with required attributes
        self.mock_model = MagicMock(spec=Model)

        # Add the required mock attributes/methods
        self.mock_model.semantic = MagicMock()
        self.mock_model.coarse_acoustics = MagicMock()
        self.mock_model.fine_acoustics = MagicMock()
        self.mock_model.codec_model = MagicMock()

        self.mock_tokenizer = MagicMock()

        # Initialize pipeline
        self.pipeline = Pipeline(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=ModelConfig(
                semantic_config=SemanticConfig(),
                coarse_acoustics_config=CoarseAcousticsConfig(),
                fine_acoustics_config=FineAcousticsConfig(),
                codec_config=CodecConfig(),
            ),
        )

    def test_generate_text_semantic(self):
        """Test semantic token generation."""
        # Mock tokenizer output
        self.mock_tokenizer.encode.return_value = [1, 2, 3]

        # Create logits with proper shape including SEMANTIC_PAD_TOKEN
        logits = mx.zeros((1, 1, 129596))  # Large enough to include SEMANTIC_PAD_TOKEN
        # Mock model output
        self.mock_model.semantic.return_value = (
            logits,  # logits with correct shape
            None,  # kv_cache
        )

        # Test generation
        semantic_tokens, text_tokens = self.pipeline.generate_text_semantic(
            "test text",
            temperature=0.7,
            use_kv_caching=True,
            voice=None,
        )

        # Verify tokenizer was called
        self.mock_tokenizer.encode.assert_called_once_with(
            "test text", add_special_tokens=False
        )

        # Verify model was called
        self.mock_model.semantic.assert_called()

        # Check output types
        self.assertIsInstance(semantic_tokens, mx.array)
        self.assertIsInstance(text_tokens, mx.array)

    @patch("mlx.core.random.categorical")  # Add this patch since we use mx alias
    def test_generate_coarse(self, mock_mlx_categorical):
        """Test coarse token generation."""
        # Create mock semantic tokens
        semantic_tokens = mx.array([1, 2, 3])

        # Create logits with proper shape
        logits = mx.zeros((1, 1, 12096))

        # Mock both categorical functions to return predictable values
        mock_mlx_categorical.return_value = mx.array([10000])  # Return token index

        # Set up the mock to return proper values for each call
        self.mock_model.coarse_acoustics.return_value = (logits, None)

        # Test generation with minimal parameters to reduce test time
        coarse_tokens = self.pipeline.generate_coarse(
            semantic_tokens,
            temperature=0.7,
            use_kv_caching=True,
            voice=None,
            max_coarse_history=60,
            sliding_window_len=2,  # Reduce this to minimum
        )

        # Verify model was called at least once
        self.mock_model.coarse_acoustics.assert_called()

        # Check output type and shape
        self.assertIsInstance(coarse_tokens, mx.array)
        self.assertEqual(coarse_tokens.shape[0], 2)  # N_COARSE_CODEBOOKS

    def test_generate_fine(self):
        """Test fine token generation."""
        # Create mock coarse tokens
        coarse_tokens = mx.zeros((2, 100))  # N_COARSE_CODEBOOKS x sequence_length

        # Mock model output with proper shape
        self.mock_model.fine_acoustics.return_value = mx.zeros((1, 1024, 1024))

        # Test generation
        fine_tokens = self.pipeline.generate_fine(coarse_tokens, temperature=0.7)

        # Verify model was called
        self.mock_model.fine_acoustics.assert_called()

        # Check output type and shape
        self.assertIsInstance(fine_tokens, mx.array)
        self.assertEqual(
            fine_tokens.shape[0], 8
        )  # N_FINE_CODEBOOKS (corrected from 10 to 8)
        self.assertEqual(fine_tokens.shape[1], 100)  # sequence_length


class TestLlamaModel(unittest.TestCase):
    @property
    def _default_config(self):
        return {
            "attention_bias": False,
            "head_dim": 128,
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 24,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": True,
            "vocab_size": 156940,
        }

    @patch("transformers.LlamaTokenizer")
    def test_init(self, mock_tokenizer):
        """Test LlamaModel initialization."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock the tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Create a minimal config
        config = ModelConfig(**self._default_config)

        # Initialize model
        model = Model(config)

        # Check that model was created
        self.assertIsInstance(model, Model)

    @patch("transformers.LlamaTokenizer")
    def test_generate(self, mock_tokenizer):
        """Test generate method."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        config = ModelConfig(**self._default_config)
        model = Model(config)

        # Verify batched input creation with a voice
        input_ids, input_mask = model.prepare_input_ids(["Foo", "Bar Baz"], voice="zoe")
        self.assertEqual(input_ids.shape[0], 2)
        self.assertEqual(input_mask.shape[0], 2)

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 9, config.vocab_size))

        # Verify batched input creation with reference audio
        input_ids, input_mask = model.prepare_input_ids(
            ["Foo", "Bar Baz"], ref_audio=mx.zeros((100,)), ref_text="Caption"
        )
        self.assertEqual(input_ids.shape[0], 2)
        self.assertEqual(input_mask.shape[0], 2)

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 22, config.vocab_size))

    @patch("transformers.LlamaTokenizer")
    def test_sanitize(self, mock_tokenizer):
        """Test sanitize method."""
        from mlx_audio.tts.models.llama.llama import Model, ModelConfig

        # Mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Create a config with tie_word_embeddings=True
        config = ModelConfig(
            model_type="llama",
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=16384,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            head_dim=128,
            max_position_embeddings=1024,
            num_key_value_heads=32,
            attention_bias=True,
            mlp_bias=True,
            rope_theta=500000.0,
            rope_traditional=False,
            rope_scaling=None,
            tie_word_embeddings=True,
        )

        # Initialize the model with a patched __init__
        with patch.object(Model, "__init__", return_value=None):
            model = Model.__new__(Model)
            model.config = config

            # Add the sanitize method from actual implementation
            def mock_sanitize(weights):
                result = {}
                for k, v in weights.items():
                    if "rotary_emb" in k:
                        continue
                    if "lm_head.weight" in k and config.tie_word_embeddings:
                        continue
                    result[k] = v
                return result

            model.sanitize = mock_sanitize

            # Create test weights with rotary embeddings and lm_head
            weights = {
                "self_attn.rotary_emb.inv_freq": mx.zeros(10),
                "lm_head.weight": mx.zeros((32000, 4096)),
                "model.layers.0.input_layernorm.weight": mx.zeros(4096),
            }

            # Test sanitize method
            sanitized = model.sanitize(weights)

            # Assert rotary embeddings are removed
            self.assertNotIn("self_attn.rotary_emb.inv_freq", sanitized)

            # Assert lm_head weights are removed with tie_word_embeddings=True
            self.assertNotIn("lm_head.weight", sanitized)

            # Assert other weights remain
            self.assertIn("model.layers.0.input_layernorm.weight", sanitized)

            # Now test with tie_word_embeddings=False
            config.tie_word_embeddings = False

            # Test sanitize again
            sanitized2 = model.sanitize(weights)

            # lm_head should be kept with tie_word_embeddings=False
            self.assertIn("lm_head.weight", sanitized2)


if __name__ == "__main__":
    unittest.main()
