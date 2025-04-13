# TTS Model Initialization Diagnostic Report

Generated: 2025-04-13 13:23:31

## Summary Statistics

- Total initialization attempts: 3
- Successful: 2 (66.7%)
- Failed: 1 (33.3%)

## Results by Model

### mlx_audio.tts.models.kokoro.kokoro.ModelConfig

- Total attempts: 2
- Success rate: 50.0%
- Average initialization time: 0.00s

#### Common Errors

- TypeError: 1 occurrences (100.0%)

### mlx_audio.tts.models.kokoro.kokoro.Model

- Total attempts: 1
- Success rate: 100.0%
- Average initialization time: 0.02s

## Detailed Failure Analysis

### mlx_audio.tts.models.kokoro.kokoro.ModelConfig - 2025-04-13 13:23:31

**Error**: TypeError: ModelConfig.__init__() missing 1 required positional argument: 'max_dur'

**Parameters**:
```json
{
  "dim_in": -1,
  "dropout": 0.1,
  "hidden_dim": 512,
  "max_conv_dim": 512,
  "multispeaker": true,
  "n_layer": 3,
  "n_mels": 80,
  "n_token": 178,
  "style_dim": 128,
  "text_encoder_kernel_size": 5,
  "istftnet": "{'upsample_kernel_sizes': [20, 12], 'upsample_rates': [10, 6], 'gen_istft_hop_size': 5, 'gen_istft_n_fft': 20, 'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 'resblock_kernel_sizes': [3, 7, 11], 'upsample_initial_channel': 512}",
  "plbert": "{'hidden_size': 768, 'num_attention_heads': 12, 'intermediate_size': 2048, 'max_position_embeddings': 512, 'num_hidden_layers': 12, 'dropout': 0.1}",
  "vocab": "{' ': 16, 'a': 43, 'b': 44, 'c': 45}"
}
```

**Initialization Trace**:
1. inspect.Parameter.__init__ in inspect.py
2. inspect.Parameter.__init__ in inspect.py
3. inspect.Parameter.__init__ in inspect.py
4. inspect.Parameter.__init__ in inspect.py
5. inspect.Parameter.__init__ in inspect.py
6. inspect.Parameter.__init__ in inspect.py
7. inspect.Parameter.__init__ in inspect.py
8. inspect.Parameter.__init__ in inspect.py
9. inspect.Parameter.__init__ in inspect.py
10. inspect.Parameter.__init__ in inspect.py
11. inspect.Parameter.__init__ in inspect.py
12. inspect.Parameter.__init__ in inspect.py
13. inspect.Parameter.__init__ in inspect.py
14. inspect.Parameter.__init__ in inspect.py
15. inspect.Parameter.__init__ in inspect.py
16. inspect.Signature.__init__ in inspect.py
17. threading.Event.__init__ in threading.py
18. threading.Condition.__init__ in threading.py
19. threading.Thread.__init__ in threading.py
20. threading.Event.__init__ in threading.py
21. threading.Condition.__init__ in threading.py
22. traceback.TracebackException.__init__ in _formatting.py
23. traceback.FrameSummary.__init__ in traceback.py
24. exceptiongroup._formatting._ExceptionPrintContext.__init__ in _formatting.py

**Traceback**:
```
Traceback (most recent call last):
  File "/Users/ZachM/Projects/mlx-audio-ui/model_diagnostics.py", line 323, in instrumented_init
    original_init(self, *args, **kwargs)
TypeError: ModelConfig.__init__() missing 1 required positional argument: 'max_dur'

```

---

