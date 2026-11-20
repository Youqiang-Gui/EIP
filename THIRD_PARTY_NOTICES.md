# Third-party notices

EIP's project license is [Apache-2.0](LICENSE). The notices below identify
upstream components; their original notices remain in the source files.

## AudioInteraction / AIM

- Source: <https://github.com/xzf-thu/Audio-Interaction>
- Model: <https://huggingface.co/zhifeixie/AudioInteraction>
- The upstream README identifies Apache-2.0 as its intended code license;
  the model card identifies Apache-2.0 for the published model.
- The `src/audiointeraction/` backbone is adapted from AudioInteraction.
  AIM model weights are downloaded separately and are not included here.

## Lightning AI / LitGPT

The following files retain Lightning AI's Apache-2.0 copyright headers:

- `src/audiointeraction/__init__.py`
- `src/audiointeraction/config.py`
- `src/audiointeraction/model.py`
- `src/audiointeraction/tokenizer.py`

The upstream license text is included in
[`licenses/Apache-2.0-Lightning-AI.txt`](licenses/Apache-2.0-Lightning-AI.txt),
from <https://github.com/Lightning-AI/litgpt/blob/v0.5.8/LICENSE>.
These files have been adapted through AudioInteraction/EIP; their retained
upstream headers do not imply that upstream authors produced the EIP changes.

## RMSNorm

The `RMSNorm` implementation in `src/audiointeraction/model.py` identifies
<https://github.com/bzhangGo/rmsnorm> as its source.
Copyright (c) 2019, Biao Zhang. The BSD 3-Clause license is included in
[`licenses/BSD-3-Clause-RMSNorm.txt`](licenses/BSD-3-Clause-RMSNorm.txt),
from <https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE>.

Other algorithm references in source comments are retained. Installed Python
dependencies and separately downloaded model/data files retain their own licenses.
