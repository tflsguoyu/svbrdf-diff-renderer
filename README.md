# MaterialGAN: Reflectance Capture using a Generative SVBRDF Model

[Yu Guo](https://tflsguoyu.github.io/), Cameron Smith, [Miloš Hašan](http://miloshasan.net/), [Kalyan Sunkavalli](http://www.kalyans.org/) and [Shuang Zhao](https://shuangz.com/).

In ACM Transactions on Graphics (SIGGRAPH Asia 2020).

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/github/teaser.jpg" width="1000px">

[[Paper](https://github.com/tflsguoyu/materialgan_paper/blob/master/materialgan.pdf)]
[[Code](https://github.com/tflsguoyu/svbrdf-diff-renderer)]
[[Supplemental Materials](https://tflsguoyu.github.io/materialgan_suppl/)]
[[Poster](https://github.com/tflsguoyu/materialgan_poster/blob/master/materialgan_poster.pdf)]
[Fastforward on Siggraph Asia 2020 ([Video](https://youtu.be/fD6CTb1DlbE))([Slides](https://www.dropbox.com/s/qi594y27dqa7irf/materialgan_ff.pptx?dl=0))] \
[Presentation on Siggraph Asia 2020 ([Video](https://youtu.be/CrAoVsJf0Zw))([Slides](https://www.dropbox.com/s/zj2mhrminoamrdg/materialgan_main.pptx?dl=0))]

## Python dependencies
numpy, torch, torchvision, opencv-python, lpips, tqdm, matplotlib, pupil_apriltags

## Usage
### Capture your own data with smartphone
1. Print "tool/tag36h11_print.png" on a solid paper with proper size and crop the center area.
2. Place it on the material you want to capture.
3. Turn on camera flashlight and capture images from different views.
4. Copy captured images to a certain folder (e.g "data/bath_tile") and run `python run_prepare.py`.
Tips:
1. All markers should be captured.
2. It's better to capture during night and turn off other lights.
2. Change camera mode to manual, keep white balance and focal lenth the same during the captures.

### Render
Run `python run_render.py`

### Optimization
Run `python run_optim.py`

