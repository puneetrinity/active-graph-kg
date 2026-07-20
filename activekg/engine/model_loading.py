from __future__ import annotations

import threading

# Transformers uses process-global state while materializing model weights. Keep
# all Hugging Face constructors serialized, even when they belong to different
# provider instances or model types.
HF_MODEL_LOAD_LOCK = threading.Lock()
