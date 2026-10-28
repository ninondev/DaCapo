# Da Capo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninondev/DaCapo/blob/main/notebook.ipynb)

[English](README.md) | 中文

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/hero_dark.svg">
  <img alt="示意图：在 ACE-Step 1.5 里，旋律经过规划器的 5 Hz 规划 token 之后，只有 29% 的帧音名正确；同一段旋律直接作为稠密音频交给渲染器，得分是 91%，和原音频（90%）一样。" src="figures/hero_light.svg" width="100%">
</picture>

我小学学了五年音乐，业余考级考到六七级（满级十级），上初中就没再碰了。后来是 AI 写歌让我又对音乐有了兴趣。我想让 AI 唱我自己写的旋律，结果歌是好听的，旋律却不是我的；想让副歌在后面再出现一次，它基本做不到。

这个仓库就是我在找原因。这里研究一个开源模型 ACE-Step 1.5，测旋律在它里面是在哪一步丢掉的。

## 谁做了什么

- 我：定了要找的问题；在 Colab 上跑 GPU 测试；决定把测试搬到 GPU 上，决定在渲染器里控制旋律；要求每个主要研究先写好规则再跑，最后的结论要用真人盲听。
- 另一个 AI：建议用 ACE-Step 1.5，给了整体计划。
- 写代码的代理：设计具体测法，写代码，跑 CPU 上的测试，算结果，写这份说明和日志。

## 模型是怎么工作的

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)（MIT 许可）分两部分。一个语言模型（规划器）先把歌写成一份粗略的规划，形式是 token，每秒 5 个。一个流匹配模型（渲染器）再根据这份规划、歌词和一句文字描述，用 8 个去噪步生成音频。渲染器也可以不用规划，直接拿一段真实音频当稠密上下文。

给旋律打分用的是音级准确率：音名（C、D、E 这些，不管八度）和乐谱对得上的时间占多少。

## 测出了什么

1. 规划会丢掉大部分旋律。经过规划器的 token 之后，只有 29% 的帧音名是对的，瞎猜大约是 10%。这还是 token 直接从这段旋律自己的音频算出来的情况，对规划器来说已经很容易了。同一段旋律直接作为稠密音频交给渲染器，得分是 91%，和原音频（90%）一样。
2. 丢失发生在 token 化这一步。用一个线性探针，从同样每秒 5 帧的音频潜变量里读音名，准确率是 0.919；从规划 token 里读，只有 0.575。
3. 在这些测试里，模型要么唱对歌词，要么唱对旋律，两样不能都对。只给歌词不给旋律，它能唱出 66% 的词。用一段没有歌词的旋律去引导它，音唱准了（86% 到 100%），歌词却掉到 3% 或更少。

所以我决定把控制放在渲染器里，因为旋律在那里还在。代码能看到 8 个去噪步中的每一步，可以读取或修改模型正在生成的东西，也可以换掉它听的文字、歌词或音频，而且不多调用一次模型。

## 结果

这里每个数字都来自 [`results/`](results/) 里的文件，每张图都由 [`figures/make_figures.py`](figures/make_figures.py) 从这些文件画出来。测试规模还小，每一节都写了样本量。

### 旋律在哪里丢失

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e1_melody_paths_dark.svg">
  <img alt="柱状图：旋律经过各条路径后的音级准确率（合并 80、120、160 BPM），shift 3.0 与 1.0：原音频 0.90 和 0.90；以稠密 25 Hz 潜变量为上下文 0.91 和 0.91；仅 VAE 往返 0.90 和 0.90；5 Hz 规划 token 0.29 和 0.31；规划 token 加新配器 0.31 和 0.32；无音频条件 0.10 和 0.11。" src="figures/e1_melody_paths_light.svg" width="100%">
</picture>

代理做了 16 条音符完全已知的旋律，每条是钢琴或长笛旋律，配和弦、贝斯和鼓，速度分别是 80、120、160 BPM。每条旋律都走一遍模型的每条路径，再测出来还剩多少旋律。

走稠密音频这条路，音名保留得和测量工具能测到的一样好。它的音符 F1 是 0.72，原音频是 0.78，差距和单纯把音频编码再解码一次差不多。走规划 token 就少多了。每个输出也和同速度的其他旋律比一次，扣掉这个随机水平之后，token 只保留了 28%（95% 置信区间 22% 到 34%）的旋律信息，音符 F1 只有 0.12。

<sub>16 条旋律 × 3 种速度，一个随机种子，NVIDIA L4，8 步 turbo 模型。shift 是采样器的一个设置：3.0 是 ACE-Step 文档给 turbo 推荐的值，默认值 1.0 结果一样。“无音频”这一项每种设置只有 6 个不同的生成结果。代理之前在 CPU 上做过一次更小的测试（每种速度 4 条旋律），90 BPM 时 token 得了 0.51，GPU 上没有重复出来，所以这里用 GPU 的数字。代码：[`experiments/e1_melody_paths`](experiments/e1_melody_paths/)。</sub>

### 为什么会丢失

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e1b_probe_dark.svg">
  <img alt="柱状图：线性探针在测试旋律上的音级准确率，规划 token 与同为 5 Hz 的音频潜变量对比：全部帧 0.575 对 0.919；帧内音高不变 0.645 对 0.982；帧内有换音 0.412 对 0.773；没学过的主奏乐器 0.504 对 0.846。多数类基线为 0.125。" src="figures/e1b_probe_light.svg" width="100%">
</picture>

帧率低不是全部原因。一个音在整个 200 毫秒的帧里都没变时，探针从音频潜变量里几乎能全读对（0.982），从规划 token 里只有 0.645。分开读每个 token 里的编码数字会好一点（0.748）。token 还把乐器的音色混进了规划：按整段算，只换主奏乐器，变掉的 token 和换一条旋律差不多多（73% 对 71% 的帧）。同一段音频编码两次，本来就会变掉大约 15%。

<sub>48 条旋律 × 3 种主奏乐器。探针在 42 条旋律上学习，在另外 6 条上测试（720 帧，一次划分）。代码：[`experiments/e1_melody_paths`](experiments/e1_melody_paths/)。</sub>

### 歌词还是旋律

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e4_tradeoff_dark.svg">
  <img alt="散点图：21 种设置（20 种引导加无引导）下，唱在乐谱音高上的音的比例和听得出的歌词比例。无引导时分别是 0.03 和 0.66。音唱准 86% 以上的设置，歌词最多只剩 0.03。378 段音频里只有 2 段两样都对了一半以上。" src="figures/e4_tradeoff_light.svg" width="100%">
</picture>

代理让模型按一条旋律唱一句歌词，每个单音节的词对一个音。没有引导时它不知道旋律，所以音只是碰巧对上（3%），但能唱出 66% 的词。然后用这段旋律的无歌词版本引导它，有两种方式：在 8 步里的前 k 步当上下文，或者加噪后当生成的起点（SDEdit，一种已有的方法，这里拿来当基线）。凡是把音唱准的设置（86% 以上），歌词都没了（3% 或更少）。378 段测试音频里，只有 2 段音和词都对了一半以上。

目前的猜测是旋律和歌词得一起引导，一个音节一个音节地对上。这就是“歌词时钟”（E6）的想法，还没做。

<sub>12 条旋律（3 条用来试设置，9 条用来测试）× 21 种设置 × 2 个随机种子。歌词由 Whisper large-v3 转写。这个测试是探索性的：设置没有在测试前登记。代码：[`experiments/e4_sing`](experiments/e4_sing/)。</sub>

### 模型会不会发展一个动机

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e5_motif_dark.svg">
  <img alt="点图，带 36 条旋律上的 95% 置信区间，0 表示随机水平。空白和模式要求的答案有多像：A B A 之后是 0.47（0.40 到 0.54），门槛 0.25；A B 之后接升 2、升 5、降 3 个半音的 A，分别是 -0.02、0.00 和 0.02，门槛 0.15；类比（A、A+5、B）是 0.01，门槛 0.15。空白和原样的 B 有多像：A B C 之后是 0.11（0.06 到 0.15）；三种移调模式里分别是 0.13、0.10 和 0.10；静音之后是 -0.03。模进 A、A+2、A+4 之后，144 次里有 0 次填的是下一步 A+6，要求是 20%。" src="figures/e5_motif_light.svg" width="100%">
</picture>

这是主要的研究（E5）。测试先放三段短旋律组成的模式，每段 4 秒，在 12 到 16 秒留一段空白，让模型用重绘（repaint）把它填上。A B A 之后，按模式应该接 B。每段空白的分数是：它的音和答案有多像，减去它和打乱后的答案有多像。打乱后的答案节奏一样，用的音也一样，所以 0 就是随机水平。假设和门槛写在[预注册文件](experiments/e5_motif/registration/prereg.json)里。第一次运行没过自己的质量检查（[`v1_outcome.json`](experiments/e5_motif/registration/v1_outcome.json)），所以这是第二次运行，开始前登记在 [addendum 2](experiments/e5_motif/registration/addendum_2.json) 里。第二次运行之前，测量工具的检查失败过一次。登记文件规定这时要重做旋律，所以我选了重做旋律，没有改检查方法（详情见 [`LOG.md`](LOG.md)）。

- 重复（H1，成立）：A B A 之后，空白对 B 的分数是 0.47（95% 置信区间 0.40 到 0.54）。登记的门槛是 0.25。
- 因为模式才重复（H2，成立）：A B C 之后，模式并不要求 B，空白对 B 的分数只有 0.11（0.06 到 0.15）。两者之差是 0.37（0.31 到 0.42），超过登记的 0.15。只看通过质量检查的空白，结论也一样（0.40，0.33 到 0.48）。所以模型重复 B，是因为模式要求它，而不只是因为 8 秒前放过 B。
- 移调（H3，不成立）：A B 之后接一个升高 2 个半音的 A，答案应该是同样升高 2 个半音的 B。空白和它对不上：升 2、升 5、降 3 个半音分别是 -0.02、0.00 和 0.02，要求是 0.15。它反而稍微更像没移调的 B（0.13、0.10 和 0.10）。
- 模进（H4，不成立）：A、A+2、A+4 之后，空白一次也没有填成下一步 A+6（144 次里 0 次，要求 20%）。
- 类比（H5，不成立）：A、A+5、B 之后，空白和 B+5 对不上：0.01（-0.02 到 0.03），要求是 0.15。

所以模型能接着它听到的模式往下走，办法是把其中一段原样重复，而重复哪一段由模式决定。移调、模进和类比都远没有达到登记的门槛。其中最大的一个效应，是降 3 个半音的移调和它的 A B C 对照相比，只有 0.03（0.005 到 0.05），而要求是 0.15。旋律都是没有人声的长笛旋律，用重绘填空，只测了一个模型。分数来自音高追踪器。关于听起来怎么样，要等真人盲听之后我才会下结论。

<sub>36 条新旋律 × 8 种条件 × 4 次抽样 = 1152 段空白，NVIDIA L4，提交 83d46e7。阶段 0b 先选了设置（重绘、长笛，112 段空白里 88 段通过，要求 84 段）。静音条件以外的 1008 段空白里，804 段通过了质量检查（要求 756 段）。按登记的规则，结论用全部抽样来定，置信区间来自对旋律的重抽样。代码：[`experiments/e5_motif`](experiments/e5_motif/)。</sub>

### 歌词注意力头跟不跟得上词

ACE-Step 1.5 用 7 个注意力头来判断每个歌词词在什么时候唱出来。读出（E6）要在歌词时钟用上它们之前，先检查它们是不是真的跟着词走。88 段无伴奏清唱里，76 段有歌声（要求 44 段），50 段能打分，也就是至少有 4 个词和 Whisper 对上（要求 20 段）。在一段典型的音频里，这些头给出的词的开始时间和 Whisper 差 0.21 秒（允许 0.25 秒）。随机挑的 7 个头的组合，中位数差 1.95 秒。50 段里有 49 段（98%，要求 75%），真正的头比典型的随机组合更准。所以按登记的规则，歌词时钟（B1）可以做。有两点要注意：在这些音频上，Whisper 和另一个对齐工具（MMS）彼此之间的中位数差就有 0.39 秒，所以 0.21 秒已经在参考答案本身的误差之内。另外，7 个头里有 6 个正好是单独最准的 6 个头，但第 7 个（第 3 层第 11 个头）单独用时差 2.41 秒。

这说明在生成好的音频里，这些头跟得上词的时间，但不说明它们能控制时间，这要由 B1 按它自己的规则来测。ACE-Step 训练时可能就让这些头和歌词对齐过，所以赢过随机的头只是一个基本检查，不算发现。

<sub>88 段音频：E4 的 12 句歌词 × 4 个种子，加上 4 段主歌 × 2 种声音 × 5 个种子，8 步 turbo 模型，NVIDIA L4。参照是 Whisper large-v3 给出的词的时间。这次读出是探索性的，它的规则只决定 B1 做不做。代码：[`experiments/e6_clock`](experiments/e6_clock/)。</sub>

### 测量本身就难

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/e0_ruler_dark.svg">
  <img alt="点图：音高追踪器在完全准确的旋律上的准确率，独奏和加响亮伴奏对比：钢琴 0.95 对 0.09，合成器 0.95 对 0.55，类人声 0.93 对 0.59。" src="figures/e0_ruler_light.svg" width="100%">
</picture>

测模型之前，代理先在每个音都已知的音频上测了测量工具。音高追踪器测独奏旋律很准，一加上响亮的伴奏就不行了。完整混音里的类人声旋律，错一两个音很难被发现（最好的 AUC 是 0.65 到 0.78）。所以有伴奏的时候，都和原音频上的同一个分数比；唱歌的部分，只给要求“无伴奏”的歌打分。

<sub>图里每种条件 9 条旋律；找错音的测试每种条件 6 条旋律。代码：[`experiments/e0_ruler`](experiments/e0_ruler/)。</sub>

## 现在在做的

钩子的代码在 [`dacapo/ace.py`](dacapo/ace.py)。在 8 个去噪步的每一步，它都能看到模型以为最后的音频会是什么样，实验可以读它、改它，或者在中途换掉文字、歌词或音频。钩子什么都不改时，输出和不加钩子完全一样（CPU 和 GPU 上差别都是 0.0），所以输出里的变化都来自钩子做的改动。

Run A 在 2026-10-28 跑完了，结果在上面。动机研究（E5）发现模型会原样重复模式里的一段，移调、模进和类比都没有达到登记的门槛。接下来是它后面的阶段：看模型的哪些层在做这个复制，以及把模型的注意力引向 B 能不能让它有意地复制。按登记文件的规定，这些阶段的设计会在运行前写进一份新的 addendum。

歌词读出（E6）过关了，所以下一步是歌词时钟（B1）。它引导那 7 个歌词注意力头，让每个词在自己的音上开始。它的规则在 Run A 之前就写进了 addendum 2：12 句歌词里至少 9 句的时间误差要减少 30% 以上，听得出的歌词比例和有歌声的音频比例各自最多下降 0.10，而且引导随机的头带来的改进要不到一半。

盲听测试用的也是这次运行的音频：60 段音频，听的人不知道每段来自哪种条件。这是我要求的。在盲听做完之前，动机研究的结论只靠音高追踪器的分数。

## 进展

| | 问题 | 状态 |
|---|---|---|
| E0 | 测量工具准不准？ | 完成 |
| E1 | 旋律经过模型每条路径后还剩多少？ | GPU 上完成：16 条旋律 × 3 种速度，两种 shift |
| E1b | 规划 token 里有什么，丢失在哪里？ | GPU 上完成：48 条旋律，一次测试划分 |
| E2 | 能不能在每个去噪步加钩子而不改变输出？ | 完成：CPU 和 GPU 上输出完全一样 |
| E3 | 能不能不管什么乐器都从音频潜变量里读出音高？ | 探索性：新乐器上 0.88（瞎猜 0.03）；专门的子空间并不比完整潜变量好 |
| E4 | 能不能既唱准音又唱清楚词？ | 探索性：12 条旋律 × 21 种设置 × 2 个种子 |
| E5 | 能不能发展一个动机？ | 第二次运行在 GPU 上完成：会把模式里的一段原样重复（H1、H2 成立），但移调、模进和类比都没有达到登记的门槛（H3 到 H5 不成立）。第一次运行没过质量检查 |
| E6 | 歌词时钟能不能对准每个音节？ | 读出在 GPU 上完成：歌词注意力头给出的词的开始时间和 Whisper 的中位数差 0.21 秒，所以可以做歌词时钟（B1），这是下一步 |
| E7 | Suno 和 ACE-Step 能把你给的旋律唱得多准？ | 已经登记，等 Suno 生成的歌 |

[`LOG.md`](LOG.md)（英文）记下了每一次运行，包括失败和犯过的错。

## 测试的规矩

先在音符已知的音频上测测量工具，再测模型。主要的研究在运行前写好假设和停止规则（这是我要求的）；有改动就加新文件，不改旧文件。不好的结果也公开。每种方法都会给同样多的尝试次数，“多生成几次挑最好的”会作为基线。不用语言模型来判断音乐，最后的结论会用真人盲听（这也是我要求的）。仓库里每个数字都能查到对应的结果文件，音频可以用固定的随机种子由代码重新生成（同一台设备结果一样，CPU 和 GPU 之间可能不同）。

## 运行

最简单的是点上面的 Colab 按钮，选 L4，点“全部运行”，再允许一次 Google Drive 访问。笔记本会自动安装、下载 ACE-Step 的开源权重（约 10 GB），然后跑 Run A：修好后的动机研究和歌词读出。在 L4 上大约 4 小时，包括安装和下载。最后你会拿到一个结果压缩包，另有一个单独的文件，里面是盲听测试的答案。运行状态存在 Drive 上（约 5 GB；E5 开始前要求至少有 7.5 GB 空间），所以运行时重置后，再点一次“全部运行”就会从停下的地方接着跑。第一次 GPU 运行的那些测试，仍然可以用 `--stages` 来跑。

在自己电脑上跑测量工具（macOS 或 Linux）：

```bash
git clone https://github.com/ninondev/DaCapo && cd DaCapo
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,transcription]"
uv pip install --no-deps basic-pitch        # 用 ONNX 模型，不装 TensorFlow
# FluidSynth 和 General MIDI 音色库：
#   Ubuntu: sudo apt install fluidsynth fluid-soundfont-gm
#   macOS:  brew install fluid-synth，然后 export DACAPO_SF2=/path/to/FluidR3_GM.sf2
pytest -q -m "not slow"                      # 单元测试；直接 pytest -q 会连端到端测试一起跑
python experiments/e0_ruler/run.py --quick   # 测试测量工具（CPU 约 3 分钟）
python figures/make_figures.py               # 从 results/ 重新画所有图
```

运行歌曲模型的测试还需要装 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)。每一步的具体命令在 [`experiments/run_all.py`](experiments/run_all.py) 里。

<details>
<summary>文件</summary>

```
dacapo/           代码库：旋律、渲染、音高追踪、打分，还有 ace.py（接到 ACE-Step 上的钩子）
experiments/      每个研究一个文件夹（e0 到 e7），还有笔记本调用的 run_all.py
results/cpu/      CPU 上跑出来的结果文件
results/gpu_run1/ 第一次 GPU 运行（NVIDIA L4）的结果文件
results/gpu_run2/ Run A 的结果文件：动机研究和歌词读出（NVIDIA L4）
figures/          图和画图的脚本
tests/            单元测试和“渲染到分析”的测试
LOG.md            按时间记下的每次运行
notebook.ipynb    Colab 笔记本
```

</details>

## 计划

- [x] 在音符已知的音频上测试测量工具
- [x] 找到旋律在开源歌曲模型里丢失的位置
- [x] 每个去噪步的钩子，在 CPU 和 GPU 上确认完全准确
- [x] 摸清音准和歌词之间的取舍（探索性）
- [x] 动机研究：修好后的运行，开始前先登记
- [x] 在 GPU 上跑修好后的动机研究和歌词读出（Run A）
- [ ] 用 Run A 的动机音频做盲听测试
- [ ] 动机研究的后续阶段：哪些层在做复制（先写新的 addendum）
- [ ] 歌词时钟（E6，阶段 B1）；读出已经过关
- [ ] 用同样的 12 条旋律测 Suno 和 ACE-Step（E7，阶段 S）
- [ ] 在渲染器里训练一个旋律适配器（E7），除非阶段 S 显示这件事已经有人做好了
- [ ] 用更多随机种子重复，再请音乐人做盲听测试
- [ ] arXiv 论文和带音频的演示页

## 致谢和许可

这个项目研究的是 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)，用到了 [SwiftF0](https://pypi.org/project/swift-f0/)、[FCPE](https://pypi.org/project/torchfcpe/)、[CREPE](https://github.com/maxrmorrison/torchcrepe)、[Basic Pitch](https://github.com/spotify/basic-pitch)、[Whisper](https://github.com/openai/whisper)、[mir_eval](https://github.com/craffel/mir_eval)、[FluidSynth](https://www.fluidsynth.org/) 和 FluidR3_GM 音色库。谢谢所有公开模型和工具的人。

引用格式：

```bibtex
@misc{dacapo2026,
  title  = {Da Capo: Measuring and Steering Musical Structure in AI Song Models},
  author = {ninondev},
  year   = {2026},
  url    = {https://github.com/ninondev/DaCapo}
}
```

MIT 许可，© 2026 ninondev（[LICENSE](LICENSE)）。
