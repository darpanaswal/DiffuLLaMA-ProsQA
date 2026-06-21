# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- DiffuGPT fix: apply the 4d-attention-mask patch in EVERY torchrun worker.
# cli.py applies it at import, but FORCE_TORCHRUN re-spawns workers that run this
# launcher directly and never import cli.py, so the patch must be applied here too.
from llamafactory import attention_patch
attention_patch.replace_attention_mask()

from llamafactory.train.tuner import run_exp  # use absolute import


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
