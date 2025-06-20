# Vocal Agent - Real-Time Speech-to-Speech AI Assistant 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Compatible-green.svg)](https://ollama.com/)

A sophisticated real-time voice assistant that seamlessly integrates speech recognition, AI reasoning, and neural text-to-speech synthesis. Designed for natural conversational interactions with advanced tool-calling capabilities.

## 🌟 Key Features

- **🎙️ Real-time Speech Processing**: Advanced speech recognition using Whisper large-v1 with Silero VAD for accurate voice activity detection
- **🧠 Intelligent Reasoning**: Powered by Llama 3.1 8B through the Agno agent framework for sophisticated AI responses
- **🌐 Web-Connected Intelligence**: Integrated web search capabilities (Google Search, Wikipedia, ArXiv) for up-to-date information
- **🗣️ Natural Voice Synthesis**: High-quality speech generation using Kokoro-82M ONNX for human-like voice output
- **⚡ Low-Latency Pipeline**: Optimized audio processing for real-time conversational experience
- **🔧 Extensible Architecture**: Modular tool system allowing easy integration of new capabilities

## 🏗️ Architecture Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Speech Recognition** | Whisper (large-v1) + Silero VAD | Convert speech to text with voice activity detection |
| **Language Model** | Llama 3.1 8B via Ollama | Natural language understanding and generation |
| **Text-to-Speech** | Kokoro-82M ONNX | Convert text responses to natural speech |
| **Agent Framework** | Agno LLM Agent | Tool orchestration and reasoning capabilities |
| **Web Integration** | Custom API connectors | Real-time information retrieval |

## 📋 Prerequisites

- **Python**: Version 3.9 or higher
- **Ollama**: Local LLM server ([Installation Guide](https://ollama.com/))
- **System Audio**: Microphone and speakers/headphones
- **Operating System**: macOS, Linux, or Windows

## 🚀 Quick Start

### 1. Install Ollama

**macOS:**
```bash
# Download from https://ollama.com/download/mac
# Or install via Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
```bash
# Download installer from https://ollama.com/download/windows
```

### 2. Clone and Setup

```bash
git clone https://github.com/danieladdisonorg/Vocal-Agent.git
cd Vocal-Agent
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip3 install -r requirements.txt
pip3 install --no-deps kokoro-onnx==0.4.7
```

### 4. Install System Dependencies

**Linux:**
```bash
sudo apt-get install espeak-ng
```

**macOS:**
```bash
brew install espeak-ng
```

**Windows:**
1. Download eSpeak NG from [releases page](https://github.com/espeak-ng/espeak-ng/releases)
2. Install the `.msi` package (e.g., `espeak-ng-20191129-b702b03-x64.msi`)

### 5. Download AI Models

**Language Model:**
```bash
ollama pull llama3.1:8b
```

**Voice Models:**
Download the following files and place them in the project root directory:
- [`kokoro-v1.0.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)
- [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)

## 🎯 Usage

### Starting the Application

1. **Start Ollama service:**
```bash
ollama serve
```

2. **Initialize the model (in a separate terminal):**
```bash
ollama run llama3.1:8b
```

3. **Launch Vocal Agent:**
```bash
python3 main.py
```

### Interaction Flow

```
🎤 Listening... Press Ctrl+C to exit
🔴 Speak now - Recording started
⏹️ Recording stopped

📝 Transcribed: "What's the weather like in Tokyo today?"
🔧 LLM Tool calls...
🤖 Response: "Let me check the current weather in Tokyo for you..."
🔊 [Audio response plays]
```

## ⚙️ Configuration

Customize the application behavior by modifying settings in `main.py`:

```python
# Audio Processing Configuration
SAMPLE_RATE = 16000          # Audio sample rate (Hz)
MAX_PHONEME_LENGTH = 500     # Maximum phoneme sequence length

# Voice Synthesis Settings
SPEED = 1.2                  # Speech rate multiplier
VOICE_PROFILE = "af_heart"   # Voice character selection

# Performance Settings
MAX_THREADS = 2              # Parallel processing threads
```

### Available Voice Profiles
- `af_heart` - Warm, friendly tone
- `af_sky` - Clear, professional tone
- `af_bella` - Expressive, dynamic tone
- Additional profiles available in `voices-v1.0.bin`

## 📁 Project Structure

```
Vocal-Agent/
├── main.py                 # Core application entry point
├── agent_client.py         # LLM agent integration layer
├── kokoro-v1.0.onnx       # Neural TTS model
├── voices-v1.0.bin        # Voice profile database
├── requirements.txt       # Python dependencies
├── vocal_agent_mac.sh     # macOS setup automation script
├── demo.png              # Application demonstration
├── LICENSE               # MIT license
└── README.md            # Project documentation
```

## 🛠️ Development

### Extending Functionality

Add new tools to the agent by integrating [Agno Toolkits](https://docs.agno.com/tools/toolkits/toolkits):

```python
from agno import Agent
from agno.tools import WebSearchTool, WikipediaSearchTool

# Add custom tools
agent = Agent(
    tools=[WebSearchTool(), WikipediaSearchTool(), YourCustomTool()],
    model="llama3.1:8b"
)
```

### Performance Optimization

- **GPU Acceleration**: Enable CUDA for faster model inference
- **Model Selection**: Choose smaller models for faster response times
- **Audio Buffer Tuning**: Adjust buffer sizes for your hardware

## 🔧 Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Ensure Ollama is running
ollama serve
# Verify model is available
ollama list
```

**Audio Device Issues:**
- Check microphone permissions
- Verify audio device selection in system settings
- Test with `python3 -c "import sounddevice; print(sounddevice.query_devices())"`

**Model Download Failures:**
- Ensure stable internet connection
- Verify sufficient disk space (models require ~8GB)
- Check Ollama service status

## 📊 Performance Metrics

- **Speech Recognition Latency**: ~200-500ms
- **LLM Response Time**: ~1-3 seconds (depending on query complexity)
- **Text-to-Speech Generation**: ~100-300ms
- **Memory Usage**: ~4-6GB (with Llama 3.1 8B)

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)** - Speech-to-text with VAD integration
- **[Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx)** - Efficient neural text-to-speech
- **[Agno](https://docs.agno.com/introduction)** - Powerful agent framework
- **[Ollama](https://ollama.ai/)** - Local LLM serving platform
- **[Weebo](https://github.com/amanvirparhar/weebo)** - Project inspiration

## 📞 Support

- **Documentation**: [Project Wiki](https://github.com/danieladdisonorg/Vocal-Agent/wiki)
- **Issues**: [GitHub Issues](https://github.com/danieladdisonorg/Vocal-Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danieladdisonorg/Vocal-Agent/discussions)

---

<div align="center">
  <strong>Built with ❤️ for the AI community</strong>
</div>
