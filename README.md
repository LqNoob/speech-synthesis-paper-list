# Content
- [speech synthesis paper list](#speech-synthesis-paper-list)
- [Zero-Shot TTS](#zero-shot-tts)
- [Other](#other)
- [Minor Points of Concern](#minor-points-of-concern)
- [Reference](#reference)


## speech synthesis paper list

- [2021/07] **SoundStream: An End-to-End Neural Audio Codec** [[paper](https://arxiv.org/abs/2107.03312)][[code](https://github.com/google/lyra)][[demo](https://google-research.github.io/seanet/soundstream/examples/)] :heavy_check_mark:
- [2022/09] **AudioLM: a Language Modeling Approach to Audio Generation** [[paper](https://arxiv.org/abs/2209.03143v2)][[demo](https://google-research.github.io/seanet/audiolm/examples/)]
- [2023/01] **InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt** [[paper](https://arxiv.org/abs/2301.13662v2)][[code](https://github.com/yangdongchao/InstructTTS)][[demo](https://dongchaoyang.top/InstructTTS/)] :heavy_check_mark:
- [2023/05] **AudioDec: An Open-source Streaming High-fidelity Neural Audio Codec** [[paper](https://arxiv.org/abs/2305.16608)][[code](https://github.com/facebookresearch/AudioDec)][[demo](https://bigpon.github.io/AudioDec_demo/)] :heavy_check_mark:
- [2023/05] **HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec** [[paper](https://arxiv.org/abs/2305.02765v2)][[code](https://github.com/yangdongchao/AcademiCodec)] *AcademiCodec* :heavy_check_mark:
- [2023/09] **High-Fidelity Audio Compression with Improved RVQGAN** [[paper](https://openreview.net/forum?id=qjnl1QUnFA)][[code](https://github.com/descriptinc/descript-audio-codec)][[demo](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5)] *DAC* :heavy_check_mark:
- [2023/09] **Soundstorm: Efficient parallel audio generation** [[paper](https://openreview.net/forum?id=KknWbD5j95)][[demo](https://google-research.github.io/seanet/soundstorm/examples/)]
- [2023/09] **High Fidelity Neural Audio Compression** [[paper](https://openreview.net/forum?id=ivCd8z8zR2)][[code](https://github.com/facebookresearch/encodec)][[code-Unofficial](https://github.com/ZhikangNiu/encodec-pytorch)] [[demo](https://ai.honu.io/papers/encodec/samples.html)] *Encodec* :heavy_check_mark:
- [2023/09] **FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec** [[paper](https://arxiv.org/abs/2309.07405v2)][[code](https://github.com/modelscope/FunCodec)][[demo](https://funcodec.github.io/)] :heavy_check_mark:
- [2023/09] **Fewer-token Neural Speech Codec with Time-invariant Codes** [[paper](https://arxiv.org/abs/2310.00014)][[code](https://github.com/y-ren16/TiCodec)][[demo](https://y-ren16.github.io/TiCodec/)] *Ti-Codec* :heavy_check_mark:
- [2023/10] **Acoustic BPE for Speech Generation with Discrete Tokens** [[paper](https://arxiv.org/abs/2310.14580)][[code](https://github.com/AbrahamSanders/codec-bpe)] :heavy_check_mark:
- [2024/01] **SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models** [[paper](https://openreview.net/forum?id=AF9Q8Vip84)][[code](https://github.com/ZhangXInFD/SpeechTokenizer)][[demo](https://0nutation.github.io/SpeechTokenizer.github.io/)] :heavy_check_mark:
- [2024/04] **SemantiCodec: An Ultra Low Bitrate Semantic Audio Codec for General Sound** [[paper](https://arxiv.org/abs/2405.00233)][[code](https://github.com/haoheliu/SemantiCodec)][[demo](https://haoheliu.github.io/SemantiCodec/)] :heavy_check_mark:
- [2024/05] **HILCodec: High Fidelity and Lightweight Neural Audio Codec** [[paper](https://arxiv.org/abs/2405.04752)][[code](https://github.com/aask1357/hilcodec)][[demo](https://aask1357.github.io/hilcodec/)] :heavy_check_mark:
- [2024/06] **Addressing Index Collapse of Large-Codebook Speech Tokenizer with Dual-Decoding Product-Quantized Variational Auto-Encoder** [[paper](https://arxiv.org/abs/2406.02940)]
- [2023/06] **UniCATS: A Unified Context-Aware Text-to-Speech Framework with Contextual VQ-Diffusion and Vocoding** [[paper](https://arxiv.org/abs/2306.07547v6)][[code](https://github.com/X-LANCE/UniCATS-CTX-vec2wav)][[demo](https://cpdu.github.io/unicats/)] :heavy_check_mark:
- [2024/04] **The X-LANCE Technical Report for Interspeech 2024 Speech Processing Using Discrete Speech Unit Challenge** [[paper](https://arxiv.org/abs/2404.06079v2)]
- [2024/06] **BiVocoder: A Bidirectional Neural Vocoder Integrating Feature Extraction and Waveform Generation** [[paper](https://arxiv.org/abs/2406.02162)][[demo](https://redmist328.github.io/BiVcoder_demo)]
- [2023/09] **Generative Pre-trained Speech Language Model with Efficient Hierarchical Transformer** [[paper](https://openreview.net/forum?id=TJNCnkDRkY)]
- [2024/06] **Spectral Codecs: Spectrogram-Based Audio Codecs for High Quality Speech Synthesis** [[paper](https://arxiv.org/abs/2406.05298)][[code](https://github.com/NVIDIA/NeMo)][[demo](https://rlangman.github.io/spectral-codec/)] :heavy_check_mark:
- [2024/01] **Finite Scalar Quantization: VQ-VAE Made Simple** [[paper](https://openreview.net/forum?id=8ishA3LxN8)][[code](https://github.com/google-research/google-research/tree/master/fsq)] *FSQ, no codebook collapse* :heavy_check_mark:
- [2024/06] **UniAudio 1.5: Large Language Model-driven Audio Codec is A Few-shot Audio Task Learner** [[paper](https://arxiv.org/abs/2406.10056)][[code](https://github.com/yangdongchao/LLM-Codec)] *LLM-Codec* :heavy_check_mark:
- [2024/04] **SNAC: Multi-Scale Neural Audio Codec** [[code](https://github.com/hubertsiuzdak/snac)][[demo](https://hubertsiuzdak.github.io/snac/)] :heavy_check_mark:
- [2023/06] **Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis** [[paper](https://arxiv.org/abs/2306.00814)][[code](https://github.com/gemelo-ai/vocos)][[demo](https://gemelo-ai.github.io/vocos/)] :heavy_check_mark:
- [2024/07] **CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens** [[paper](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf)][[code](https://github.com/FunAudioLLM/CosyVoice)][[demo](https://fun-audio-llm.github.io/)] :heavy_check_mark:
- [2024/06] **Single-Codec: Single-Codebook Speech Codec towards High-Performance Speech Generation** [[paper](https://www.arxiv.org/abs/2406.07422)][[demo](https://kkksuper.github.io/Single-Codec/)]
- [2024/02] **APCodec: A Neural Audio Codec with Parallel Amplitude and Phase Spectrum Encoding and Decoding** [[paper](https://arxiv.org/abs/2402.10533)][[code](https://github.com/YangAi520/APCodec)] *code comming soon*
- [2024/07] **dMel: Speech Tokenization made Simple** [[paper](https://arxiv.org/abs/2407.15835)] *code comming soon*
- [2024/07] **SuperCodec: A Neural Speech Codec with Selective Back-Projection Network** [[paper](https://arxiv.org/abs/2407.20530)][[code](https://github.com/exercise-book-yq/Supercodec)][[demo](https://exercise-book-yq.github.io/SuperCodec-Demo/)] :heavy_check_mark:
- [2024/04] **ESC: Efficient Speech Coding with Cross-Scale Residual Vector Quantized Transformers** [[paper](https://arxiv.org/abs/2404.19441)][[code](https://github.com/yzGuu830/efficient-speech-codec)] :heavy_check_mark:
- [2024/02] **Language-Codec: Reducing the Gaps Between Discrete Codec Representation and Speech Language Models** [[paper](https://arxiv.org/abs/2402.12208)][[code](https://github.com/jishengpeng/Languagecodec)][[demo](https://languagecodec.github.io/)] :heavy_check_mark:
- [2024/06] **SimpleSpeech: Towards Simple and Efficient Text-to-Speech with Scalar Latent Transformer Diffusion Models** [[paper](https://arxiv.org/abs/2406.02328v2)][[code](https://github.com/yangdongchao/SimpleSpeech)][[demo](https://simplespeech.github.io/simplespeechDemo/)] *SQ-Codec* | *Code Comming Soon*
- [2024/08] **SimpleSpeech 2: Towards Simple and Efficient Text-to-Speech with Flow-based Scalar Latent Transformer Diffusion Models** [[paper](https://arxiv.org/abs/2408.13893)][[demo](https://dongchaoyang.top/SimpleSpeech2_demo/)]
- [2024/08] **Music2Latent: Consistency Autoencoders for Latent Audio Compression** [[paper](https://www.arxiv.org/abs/2408.06500)][[code](https://github.com/SonyCSLParis/music2latent)][[demo](https://sonycslparis.github.io/music2latent-companion/)] *continuous latent space* :heavy_check_mark:
- [2024/08] **WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling** [[paper](https://arxiv.org/abs/2408.16532)][[code](https://github.com/jishengpeng/WavTokenizer)][[demo](https://wavtokenizer.github.io/)] :heavy_check_mark:
- [2024/08] **Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model** [[paper](https://arxiv.org/abs/2408.17175)][[code](https://github.com/zhenye234/xcodec)][[demo](https://x-codec-audio.github.io/)] *X-Codec* :heavy_check_mark:
- [2024/09] **SoCodec: A Semantic-Ordered Multi-Stream Speech Codec for Efficient Language Model Based Text-to-Speech Synthesis** [[paper](https://arxiv.org/abs/2409.00933)][[code](https://github.com/hhguo/SoCodec)][[demo](https://hhguo.github.io/DemoSoCodec/)] :heavy_check_mark:
- [2024/09] **Speaking from Coarse to Fine: Improving Neural Codec Language Model via Multi-Scale Speech Coding and Generation** [[paper](https://arxiv.org/abs/2409.11630v1)][[demo](https://hhguo.github.io/DemoCoFiSpeech/)] *CoFi-Speech*
- [2024/09] **NDVQ: Robust Neural Audio Codec with Normal Distribution-Based Vector Quantization** [[paper](https://arxiv.org/abs/2409.12717)]

## Zero-Shot TTS

- [2023/05] **Better speech synthesis through scaling** [[paper](https://arxiv.org/abs/2305.07243)][[code](https://github.com/neonbjb/tortoise-tts)] *tortoise-tts* :heavy_check_mark:
- [2023/09] **Voiceflow: Efficient text-to-speech with rectified flow matching** [[paper](https://arxiv.org/abs/2309.05027v2)][[code](https://github.com/X-LANCE/VoiceFlow-TTS)][[demo](https://cantabile-kwok.github.io/VoiceFlow/)] :heavy_check_mark:
- [2023/09] **Voicebox: Text-guided multilingual universal speech generation at scale** [[paper](https://openreview.net/forum?id=gzCS252hCO)][[demo](https://voicebox.metademolab.com/)]
- [2023/09] **Matcha-tts: A fast tts architecture with conditional flow matching** [[paper](https://arxiv.org/abs/2309.03199v2)][[code](https://github.com/shivammehta25/Matcha-TTS)][[demo](https://shivammehta25.github.io/Matcha-TTS/)] :heavy_check_mark:
- [2023/01] **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers** [[paper](https://arxiv.org/abs/2301.02111v1)][[code](https://github.com/microsoft/unilm)][[demo](https://www.microsoft.com/en-us/research/project/vall-e-x/)] *VALL-E* :heavy_check_mark:
- [2024/03] **VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild** [[paper](https://arxiv.org/abs/2403.16973v2)][[code](https://github.com/jasonppy/VoiceCraft)][[demo](https://jasonppy.github.io/VoiceCraft_web/)] :heavy_check_mark:
- [2024/01] **NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers** [[paper](https://openreview.net/forum?id=Rc7dAwVL3v)][[demo](https://speechresearch.github.io/naturalspeech2/)]
- [2024/03] **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models** [[paper](https://arxiv.org/abs/2403.03100v3)][[demo](https://speechresearch.github.io/naturalspeech3/)]
- [2024/01] **Mega-TTS 2: Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis** [[paper](https://openreview.net/forum?id=mvMI3N4AvD)][[demo](https://boostprompt.github.io/boostprompt/)]
- [2024/03] **HAM-TTS: Hierarchical Acoustic Modeling for Token-Based Zero-Shot Text-to-Speech with Model and Data Scaling** [[paper](https://arxiv.org/abs/2403.05989)][[demo](https://anonymous.4open.science/w/ham-tts/)]
- [2024/06] **ControlSpeech: Towards Simultaneous Zero-shot Speaker Cloning and Zero-shot Language Style Control With Decoupled Codec** [[paper](https://arxiv.org/abs/2406.01205)][[code](https://github.com/jishengpeng/ControlSpeech)][[demo](https://controlspeech.github.io/)] :heavy_check_mark:
- [2024/06] **XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model** [[paper](https://arxiv.org/abs/2406.04904)][[code](https://github.com/coqui-ai/TTS/tree/main)][[demo](https://edresson.github.io/XTTS/)] :heavy_check_mark:
- [2024/06] **VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers** [[paper](https://arxiv.org/abs/2406.05370)][[demo](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/)]
- [2024/06] **Autoregressive Diffusion Transformer for Text-to-Speech Synthesis** [[paper](https://www.arxiv.org/abs/2406.05551)][[demo](https://ardit-tts.github.io/)]
- [2024/06] **VALL-E R: Robust and Efficient Zero-Shot Text-to-Speech Synthesis via Monotonic Alignment** [[paper](https://arxiv.org/abs/2406.07855)][[demo](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-r/)]
- [2024/06] **DiTTo-TTS: Efficient and Scalable Zero-Shot Text-to-Speech with Diffusion Transformer** [[paper](https://arxiv.org/abs/2406.11427)][[demo](https://ditto-tts.github.io/)]
- [2024/01] **CLaM-TTS: Improving Neural Codec Language Model for Zero-Shot Text-to-Speech** [[paper](https://openreview.net/forum?id=ofzeypWosV)][[demo](https://clam-tts.github.io/)]
- [2024/06] **TacoLM: GaTed Attention Equipped Codec Language Model are Efficient Zero-Shot Text to Speech Synthesizers** [[paper](https://arxiv.org/abs/2406.15752)][[code](https://github.com/Ereboas/TacoLM)][[demo](https://ereboas.github.io/TacoLM/)] :heavy_check_mark:
- [2023/11] **HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-shot Speech Synthesis** [[paper](https://arxiv.org/abs/2311.12454)][[code](https://github.com/sh-lee-prml/HierSpeechpp)][[demo](https://sh-lee-prml.github.io/HierSpeechpp-demo/)] :heavy_check_mark:
- [2024/06] **E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS** [[paper](https://arxiv.org/abs/2406.18009)][[demo](https://www.microsoft.com/en-us/research/project/e2-tts/)] *similar to Seed-TTS*
- [2024/07] **Robust Zero-Shot Text-to-Speech Synthesis with Reverse Inference Optimization** [[paper](https://arxiv.org/abs/2407.02243)][[demo](https://yuchen005.github.io/RIO-TTS-demos/)] *Human FeedBack*
- [2024/07] **CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens** [[paper](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf)] [[code](https://github.com/FunAudioLLM/CosyVoice)][[demo](https://funaudiollm.github.io/)] :heavy_check_mark:
- [2024/04] **FlashSpeech: Efficient Zero-Shot Speech Synthesis** [[paper](https://arxiv.org/abs/2404.14700)][[code](https://github.com/zhenye234/FlashSpeech)][[demo](https://flashspeech.github.io/)] *code comming soon*
- [2024/08] **Bailing-TTS: Chinese Dialectal Speech Synthesis Towards Human-like Spontaneous Representation** [[paper](https://arxiv.org/abs/2408.00284)][[demo](https://c9412600.github.io/bltts_tech_report/index.html)]
- [2024/08] **VoxInstruct: Expressive Human Instruction-to-Speech Generation with Unified Multilingual Codec Language Modelling** [[paper](https://www.arxiv.org/abs/2408.15676)][[code](https://github.com/thuhcsi/VoxInstruct)][[demo](https://voxinstruct.github.io/VoxInstruct/)] *code comming soon*
- [2024/09] **MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer** [[paper](https://arxiv.org/abs/2409.00750)][[demo](https://maskgct.github.io/)] *Masked Generative Model* | *Similar to Seed-TTS*
- [2024/09] **FireRedTTS: A Foundation Text-To-Speech Framework for Industry-Level Generative Speech Applications** [[paper](https://www.arxiv.org/abs/2409.03283)][[demo](https://fireredteam.github.io/demos/firered_tts/)] *voice cloning for dubbing and human-like speech generation for chatbots*
- [2024/09] **Takin: A Cohort of Superior Quality Zero-shot Speech Generation Models** [[paper](https://arxiv.org/abs/2409.12139)][[demo](https://takinaudiollm.github.io/)]

## Other

- [2024/05] **Instruct-MusicGen: Unlocking Text-to-Music Editing for Music Language Models via Instruction Tuning** [[paper](https://arxiv.org/abs/2405.18386v2)][[code](https://github.com/ldzhangyx/instruct-MusicGen)][[demo](https://foul-ice-5ea.notion.site/Instruct-MusicGen-Demo-Page-a1e7d8d474f74df18bda9539d96687ab)] *Instruction Tuning* :heavy_check_mark:
- [2024/04] **StoryTTS: A Highly Expressive Text-to-Speech Dataset with Rich Textual Expressiveness Annotations** [[paper](https://arxiv.org/abs/2404.14946)][[code](https://github.com/X-LANCE/StoryTTS)][[demo](https://goarsenal.github.io/StoryTTS/)] *Lian Liru(连丽如) dataset* :heavy_check_mark:
- [2024/04] **CoVoMix: Advancing Zero-Shot Speech Generation for Human-like Multi-talker Conversations** [[paper](https://arxiv.org/abs/2404.06690)][[demo](https://www.microsoft.com/en-us/research/project/covomix/)] *multi-round dialogue speech generation*
- [2024/04] **SpeechAlign: Aligning Speech Generation to Human Preferences** [[paper](https://arxiv.org/abs/2404.05600)][[code](https://github.com/0nutation/SpeechGPT)][[demo](https://0nutation.github.io/SpeechAlign.github.io/)] *Human Feedback* :heavy_check_mark:
- [2024/06] **Enhancing Zero-shot Text-to-Speech Synthesis with Human Feedback** [[paper](https://www.arxiv.org/abs/2406.00654)] *Human Feedback*
- [2024/06] **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** [[paper](https://arxiv.org/abs/2406.02430)][[demo](https://bytedancespeech.github.io/seedtts_tech_report/)]
- [2024/06] **WenetSpeech4TTS: A 12,800-hour Mandarin TTS Corpus for Large Speech Generation Model Benchmark** [[paper](https://arxiv.org/abs/2406.05763v2)][[demo](https://huggingface.co/Wenetspeech4TTS)] 
- [2024/02] **Natural language guidance of high-fidelity text-to-speech with synthetic annotations** [[paper](https://arxiv.org/abs/2402.01912)][[code](https://github.com/huggingface/parler-tts)][[demo](https://www.text-description-to-speech.com/)] *Prompt Control | Parler-TTS* :heavy_check_mark:
- [2023/06] **Simple and Controllable Music Generation** [[paper](https://arxiv.org/abs/2306.05284)][[code](https://github.com/facebookresearch/audiocraft)] *Prompt Control | AudioCraft* :heavy_check_mark:
- [2023/02] **Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision** [[paper](https://arxiv.org/abs/2302.03540)][[code](https://github.com/collabora/WhisperSpeech)][[demo](https://collabora.github.io/WhisperSpeech/)] *SpearTTS | WhisperSpeech* :heavy_check_mark:
- [2024/06] **High Fidelity Text-to-Speech Via Discrete Tokens Using Token Transducer and Group Masked Language Model** [[paper](https://arxiv.org/abs/2406.17310)][[demo](https://srtts.github.io/interpreting-speaking/)] *Transducer/End-to-End*
- [2024/01] **VALL-T: Decoder-Only Generative Transducer for Robust and Decoding-Controllable Text-to-Speech** [[paper](https://arxiv.org/abs/2401.14321)][[code](https://github.com/cpdu/vallt)][[demo](https://cpdu.github.io/vallt/)] *code comming soon | Transducer*
- [2024/01] **Utilizing Neural Transducers for Two-Stage Text-to-Speech via Semantic Token Prediction** [[paper](https://arxiv.org/abs/2401.01498)][[demo](https://gannnn123.github.io/token-transducer/)] *Transducer/End-to-End*
- [2024/06] **Improving Robustness of LLM-based Speech Synthesis by Learning Monotonic Alignment** [[paper](https://arxiv.org/abs/2406.17957v1)][[demo](https://t5tts.github.io/)] *Monotonic Alignment*
- [2024/01] **EmotiVoice 😊: a Multi-Voice and Prompt-Controlled TTS Engine** [[code](https://github.com/netease-youdao/EmotiVoice)] :heavy_check_mark:
- [2024/07] **Spontaneous Style Text-to-Speech Synthesis with Controllable Spontaneous Behaviors Based on Language Models** [[paper](https://arxiv.org/abs/2407.13509)][[demo](https://thuhcsi.github.io/interspeech2024-SponLMTTS/)] *Spontaneous*
- [2024/07] **Stable Audio Open** [[paper](https://arxiv.org/abs/2407.14358)] [[code](https://huggingface.co/stabilityai/stable-audio-open-1.0)] :heavy_check_mark:
- [2024/02] **Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities** [[paper](https://arxiv.org/abs/2402.01831)][[code](https://github.com/NVIDIA/audio-flamingo)][[demo](https://audioflamingo.github.io/)] :heavy_check_mark:
- [2024/05] **EmoLLM(心理健康大模型)** [[code](https://github.com/SmartFlowAI/EmoLLM/blob/main/generate_data/tutorial.md)][[demo](https://openxlab.org.cn/apps/detail/Farewell1/EmoLLMV2.0)] :heavy_check_mark:
- [2024/08] **EELE: Exploring Efficient and Extensible LoRA Integration in Emotional Text-to-Speech** [[paper](https://www.arxiv.org/abs/2408.10852)] *LORA*
- [2024/08] **StyleSpeech: Parameter-efficient Fine Tuning for Pre-trained Controllable Text-to-Speech** [[paper](https://www.arxiv.org/abs/2408.14713)][[demo](https://style-speech.vercel.app/)] *LORA*
- [2024/08] **VoiceTailor: Lightweight Plug-In Adapter for Diffusion-Based Personalized Text-to-Speech** [[paper](https://arxiv.org/abs/2408.14739)][[demo](https://voicetailor.github.io/)] *LORA*
- [2024/08] **SSL-TTS: Leveraging Self-Supervised Embeddings and kNN Retrieval for Zero-Shot Multi-speaker TTS** [[paper](https://www.arxiv.org/abs/2408.10771)][[demo](https://www.arxiv.org/abs/2408.10771)] *SSL*
- [2024/08] **Style-Talker: Finetuning Audio Language Model and StyleBased Text-to-Speech Model for Fast Spoken Dialogue Generation** [[paper](https://arxiv.org/abs/2408.11849)][[code](https://github.com/xi-j/Style-Talker)][[demo](https://styletalker.github.io/)] *code comming soon*
- [2024/08] **DualSpeech: Enhancing Speaker-Fidelity and Text-Intelligibility Through Dual Classifier-Free Guidance** [[paper](https://arxiv.org/abs/2408.14423)][[demo](https://dualspeech.notion.site/DualSpeech-Demo-25fcf06ea94b4a739094d828d400542d)]
- [2023/10] **SALMONN: Towards Generic Hearing Abilities for Large Language Models** [[paper](https://arxiv.org/abs/2310.13289)][[code](https://github.com/bytedance/SALMONN)] :heavy_check_mark:
- [2024/03] **WavLLM: Towards Robust and Adaptive Speech Large Language Model** [[paper](https://arxiv.org/abs/2404.00656)][[code](https://github.com/microsoft/SpeechT5/tree/main/WavLLM)] :heavy_check_mark:
- [2024/02] **AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling** [[paper](https://arxiv.org/abs/2402.12226)][[code](https://github.com/OpenMOSS/AnyGPT)][[demo](https://junzhan2000.github.io/AnyGPT.github.io/)] :heavy_check_mark:
- [2023/11] **Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models** [[paper](https://arxiv.org/abs/2311.07919)][[code](https://github.com/QwenLM/Qwen-Audio)] *speech interaction model* :heavy_check_mark:
- [2024/07] **Qwen2-Audio Technical Report** [[paper](https://www.arxiv.org/abs/2407.10759)][[code](https://github.com/QwenLM/Qwen2-Audio)] *speech interaction model* :heavy_check_mark:
- [2024/07] **Speech-Copilot: Leveraging Large Language Models for Speech Processing via Task Decomposition, Modularization, and Program Generation** [[paper](https://arxiv.org/abs/2407.09886)][[code](https://github.com/kuan2jiu99)][[demo](https://sites.google.com/view/slt2024-demo-page)] *code coming soon | speech interaction model*
- [2024/06] **GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities** [[paper](https://arxiv.org/abs/2406.11768)][[code](https://github.com/Sreyan88/GAMA)][[demo](https://sreyan88.github.io/gamaaudio/)] :heavy_check_mark:
- [2024/07] **Generative Expressive Conversational Speech Synthesis** [[paper](https://arxiv.org/abs/2407.21491)][[code](https://github.com/walker-hyf/GPT-Talker)] *GPT-Talker | code comming soon*
- [????/??] **SpeechGPT2: End-to-End Human-Like Spoken Chatbot** [[paper]()][[code](https://github.com/0nutation/SpeechGPT)][[demo](https://0nutation.github.io/SpeechGPT2.github.io/)] *paper & code comming soon | speech interaction model*
- [2024/08] **Language Model Can Listen While Speaking** [[paper](https://arxiv.org/abs/2408.02622)][[demo](https://ziyang.tech/LSLM/)] *Full Duplex Modeling | speech interaction model*
- [2024/08] **VITA: Towards Open-Source Interactive Omni Multimodal LLM** [[paper](https://www.arxiv.org/abs/2408.05211)][[code](https://github.com/VITA-MLLM/VITA)][[demo](https://vita-home.github.io/)] *code comming soon | speech interaction model*
- [2024/08] **Speech To Speech: an effort for an open-sourced and modular GPT4-o** [[code](https://github.com/huggingface/speech-to-speech)] *End-to-End | speech interaction model* :heavy_check_mark:
- [2024/08] **Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming** [[paper](https://arxiv.org/abs/2408.16725)][[code](https://github.com/gpt-omni/mini-omni)] *End-to-End | speech interaction model* :heavy_check_mark:
- [2024/09] **FLUX that Plays Music** [[paper](https://arxiv.org/abs/2409.00587)][[code](https://github.com/feizc/FluxMusic)][[melodio](https://www.melodio.ai/)] *music generation | KunLun* :heavy_check_mark:
 
## Minor Points of Concern

<details>
<summary>GitHub</summary>
 
- ChatTTS: https://github.com/2noise/ChatTTS/tree/main
- OpenVoice: https://github.com/myshell-ai/OpenVoice
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- VoiceCraft: https://github.com/jasonppy/VoiceCraft
- YourTTS: https://github.com/Edresson/YourTTS
- Coqui: https://github.com/coqui-ai/TTS
- MARS5-TTS: https://github.com/Camb-ai/MARS5-TTS
- edge-tts: https://github.com/rany2/edge-tts
- metavoice-src: https://github.com/metavoiceio/metavoice-src
- StyleTTS2: https://github.com/yl4579/StyleTTS2
- open-tts-tracker: https://github.com/Vaibhavs10/open-tts-tracker
- Amphion: https://github.com/open-mmlab/Amphion
- CTranslate2: https://github.com/OpenNMT/CTranslate2
- CFM: https://github.com/atong01/conditional-flow-matching
- speech-trident: https://github.com/ga642381/speech-trident
- bark: https://github.com/suno-ai/bark
- LangGPT: https://github.com/langgptai/LangGPT (提示词工程)
- composio: https://github.com/ComposioHQ/composio
</details>

<details>
<summary>Nice Tool</summary>
 
- pytorch-OpCounter: https://github.com/Lyken17/pytorch-OpCounter
- rich: https://github.com/Textualize/rich
- argbind: https://github.com/pseeth/argbind/
- audiotools: https://github.com/descriptinc/audiotools
- hydra: https://github.com/facebookresearch/hydra
- joblib: https://github.com/joblib/joblib
- einops: https://github.com/arogozhnikov/einops
- safetensors: https://github.com/huggingface/safetensors
- OpenDiloco: https://github.com/PrimeIntellect-ai/OpenDiloco
- WeTextProcessing: https://github.com/wenet-e2e/WeTextProcessing
- zed: https://github.com/zed-industries/zed
- weekly: https://github.com/ljinkai/weekly
- tinygrad: https://github.com/tinygrad/tinygrad
- ffmpeg-normalize: https://github.com/slhck/ffmpeg-normalize
- kohya_ss: https://github.com/bmaltais/kohya_ss
- Lora-Training-in-Comfy: https://github.com/LarryJane491/Lora-Training-in-Comfy
- ComfyUI-Manager: https://github.com/ltdrdata/ComfyUI-Manager
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- comfyui-workspace-manager: https://github.com/11cafe/comfyui-workspace-manager
- CosyVoice+ComfyUI: https://github.com/AIFSH/CosyVoice-ComfyUI
- ComfyUI-wiki: https://github.com/602387193c/ComfyUI-wiki
- ZHO: https://github.com/ZHO-ZHO-ZHO
- tmux: https://github.com/tmux/tmux
- LoRAlib: https://github.com/microsoft/LoRA
- codespaces: https://github.com/codespaces
- Foliate(PDF): https://johnfactotum.github.io/foliate/
- Okular(PDF): https://okular.kde.org/zh-cn/
</details>

## Reference

- [别慌！一文教你看懂GPT-4o背后的语音技术](https://zhuanlan.zhihu.com/p/698725358)
- [百花齐放的Audio Codec: 语音合成利器](https://zhuanlan.zhihu.com/p/696434090)
- [InterSpeech2024 Speech Processing Using Discrete Speech Units](https://interspeech2024.org/special-sessions-challenges/) : https://www.wavlab.org/activities/2024/Interspeech2024-Discrete-Speech-Unit-Challenge/ : https://huggingface.co/discrete-speech : [arxiv 2024](https://arxiv.org/abs/2406.07725)
- [Towards audio language modeling -- an overview](https://arxiv.org/abs/2402.13236)
- [Codec-SUPERB: An In-Depth Analysis of Sound Codec Models](https://arxiv.org/abs/2402.13071v2) : https://github.com/voidful/Codec-SUPERB
- [EMO-Codec: A Depth Look at Emotion Preservation Capacity of Legacy and Neural Codec Models With Subjective and Objective Evaluations](https://arxiv.org/abs/2407.15458)



