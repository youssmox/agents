# Copyright 2023 LiveKit, Inc.
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

from __future__ import annotations

from enum import Enum
from typing import Literal


class RealtimeModels(str, Enum):
    """Available models for ElevenLabs realtime transcription"""

    SCRIBE_V2_REALTIME = "scribe_v2_realtime"


class AudioFormat(str, Enum):
    """Audio format options for realtime transcription"""

    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_44100 = "pcm_44100"


class CommitStrategy(str, Enum):
    """Strategy for committing transcription results"""

    VAD = "vad"
    MANUAL = "manual"
