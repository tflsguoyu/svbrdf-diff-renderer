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
numpy, torch, torchvision, opencv-python, lpips, tqdm, matplotlib, pupil_apriltags, pillow-heif

## Pretrained MaterialGAN model
Download all the checkpoints to `ckp`: 
[`materialgan.pth`](https://www.dropbox.com/scl/fi/z41e6tedyh7m57vatse7p/materialgan.pth)
[`latent_avg_W+_256.pt`](https://www.dropbox.com/scl/fi/nf4kfoiqx6h7baxpbfu01/latent_avg_W-_256.pt)
[`latent_const_W+_256.pt`](https://www.dropbox.com/scl/fi/mdh8boshpfc6lwktrfh4i/latent_const_W-_256.pt)
[`latent_const_N_256.pt`](https://www.dropbox.com/scl/fi/320aov4ahc4wkhaq8mpve/latent_const_N_256.pt)
[`vgg_conv.pt`](https://www.dropbox.com/scl/fi/hp8bxxyejkw7d9a9gxxhc/vgg_conv.pt)

## Generate target images for optimization
To get an optimized SVBRDF maps, we need target images and a corresponding json file. 
We provide 3 kinds of target images generation methods here,

### Random textures
You can generate random textures by using MaterialGAN.
1. Create a data folder, e.g. `data/random`, and `target.json` file in it. You can define how many target images will be created (`idx`) and their coresponding camera/light positions (`camera_pos` and `light_pos`).  
2. Run `python run_gentextures.py`.
3. The generated SVBRDF maps are in `data/random/target/tex/256` and target images are in `data/random/target/img/256`

### Existing textures
You can use synthetic SVBRDF maps from 3rd party.
1. Create a data folder, e.g. `data/card_blue`, and `target.json` file in it similar to previous one.
2. Rename and copy SVBRDF maps to `data/card_blue/target/tex/256`. It should contain 4 maps, `dif.png`: diffuse albedo; `nom.png`: normal map; `rgh.png`: roughness; `spe.png`: specular albedo. You can use `tex4to1()` in `run_unittest.py` to combine 4 images, but not necessary.
3. Run `python run_render.py`.
4. The generated target images are in `data/card_blue/target/img/256`

### Capture your own data with smartphone
1. Print "fig/tag36h11_print.png" on a solid paper with a proper size and crop the center area.
2. Measure `size`(in cm unit) with a ruler, see the red arrow line in below figure.
3. Place it on the material you want to capture, and make the paper as flat as possible.
4. Turn on camera flashlight and capture images from different views.
5. Create a data folder, e.g `data/yellow_box`, and copy captured images to `data/yellow_box/raw`.
6. Run `python run_prepare.py`. 
The `size` here in `input_obj.eval(size=17, depth=0.1)` is the number you measured from step 2. `depth` is distance (in cm unit) between marker plane and material plane. For example, if you attach the markers on a thick cardboard, you should use a larger `depth`.
7. The generate target images is located in `data/yellow_box/target/img/1024` and corresponding `optim.json` file is generated as well.
<img src="https://github.com/tflsguoyu/svbrdf-diff-renderer/blob/master/fig/fig1.png" width="600px">

Tips:
1. All markers should be captured and in focus and the letter `A` should be facing up.
2. It's better to capture during the night or in a dark room and turn off other lights.
3. It's better to see the highlights in the cropped area.
4. Change camera mode to manual, keep white balance and focal lenth the same during the captures.
5. `.HEIC` image format is not supported now. Convert it to `.PNG` first. 
6. Preferred capturing order: highlight in topleft -> top -> topright -> left -> center -> right -> bottomleft -> bottom -> bottomright. See images in `data/yellow_box/raw` as references.

## Optimization
After the target images are generated, and before optimization starts, the `optim.json` should be set up well in the data folder you want ot process.  

### Per-pixel optimization on SVBRDF maps, which is considered as a baseline.
Run `python run_optim_svbrdf.py`

### Optimization on MaterialGAN latent space (TODO, still working on it)
Run `python run_optim_ganlatent.py`

## Notes
- 01/21/2024: TODO: latent space optimization
- 12/30/2023: Start to use this repo.
