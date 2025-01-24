# audiologger
![alt text](https://github.com/sciapponi/audiologger/blob/main/img/diagram-20241121.png)
# General Information
Audio Logger with VAD and Neural Audio Codec on RaspberryPi and MQTT
# RaspberryPi

Clone this Repository:
```bash
git clone https://github.com/sciapponi/audiologger
```
Setup or replace the config file, which has the following structure:
```yaml
LOGGER_ID: 1
SAMPLING_RATE: 16000
CHUNK_SIZE: 512

SILERO:
  MODEL_PATH: "models/silero_vad.onnx"

NAC:
  MODEL_PATH: "models/encoder_quant_fp16.onnx"

MQTT:
  BROKER_ADDRESS: "localhost"
```

Run the **main.py** script with optional overriding of the config file name by:

```bash
python3 main.py --config-name=your-config-name.yaml
```
# Transcriber Client Setup
### NVIDIA JETSON SETUP

Clone Jetson Containers repo:
```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```
Start whisper container
```bash
jetson-containers run $(autotag whisper)
```
Clone this repo inside the Whisper Container
```bash
cd ~
git clone https://github.com/sciapponi/audiologger
```

Install the Transcriber Client Requirements, edit or create a new config file and run the **main.py** to start the client.

The config file has the following structure:

```yaml
SAMPLING_RATE: 16000

NAC:
  MODEL_PATH: "/home/ste/Code/audiologger/TranscribeClient/model/quant_decoder_components.ckpt"

MQTT:
  BROKER_ADDRESS: "localhost"

WHISPER:
  LANGUAGE: "en"
```

If a new config file is created inside the **config** folder, it can be specified as an argument via

```
python3 main.py --config-name=your-config-name.yaml
```
### Dealing with SoundStream Compatibility

If the Jetson model only supports python<3.10 (due to JetPack being version <=5) it won't be able to install SoundStream trough PIP, to get around it:

Clone the following repository inside the **TranscribeClient** folder:
```
git clone https://github.com/sciapponi/soundstream_fork
```

replace the following line in the **TranscribeClient/model/decoder.py** file:

```python
import torch
from torch import nn
from vector_quantize_pytorch import ResidualVQ
# from soundstream.decoder import Decoder as SoundStreamDecoder
from soundstream_fork.decoder import Decoder as SoundStreamDecoder
```




