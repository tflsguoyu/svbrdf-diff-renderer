# MaterialGAN: Reflectance Capture using a Generative SVBRDF Model

[Yu Guo](https://tflsguoyu.github.io/), Cameron Smith, [Miloš Hašan](http://miloshasan.net/), [Kalyan Sunkavalli](http://www.kalyans.org/) and [Shuang Zhao](https://shuangz.com/).

In ACM Transactions on Graphics (SIGGRAPH Asia 2020).

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/github/teaser.jpg" width="600px">

[[Paper](https://github.com/tflsguoyu/materialgan_paper/blob/master/materialgan.pdf)]
[[Code](https://github.com/tflsguoyu/svbrdf-diff-renderer)]
[[Supplemental Materials](https://tflsguoyu.github.io/materialgan_suppl/)]
[[Poster](https://github.com/tflsguoyu/materialgan_poster/blob/master/materialgan_poster.pdf)]
[Fastforward on Siggraph Asia 2020 ([Video](https://youtu.be/fD6CTb1DlbE))([Slides](https://www.dropbox.com/s/qi594y27dqa7irf/materialgan_ff.pptx?dl=0))]
[Presentation on Siggraph Asia 2020 ([Video](https://youtu.be/CrAoVsJf0Zw))([Slides](https://www.dropbox.com/s/zj2mhrminoamrdg/materialgan_main.pptx?dl=0))]
[[Dataset(38)](https://drive.google.com/file/d/1Vs2e35c4bNHRUu3ON4IsuOOP6uK8Ivji/view?usp=sharing)]
[[Dataset_Zhou(76)](https://drive.google.com/file/d/1kfefC6YbkbSazLeJ7uUUUFR6WEeWozgA/view?usp=sharing)]

# Quick start

## 1. Python dependencies (prefer to use `pip install`)

`torch`, `torchvision`, `opencv-python`, `matplotlib`, `tqdm`, `pupil-apriltags`(for data capture), `mitsuba`(for envmap rendering)

Tested on,

1. MacOS, python3.11, pytorch2.5.1, CPU
2. Windows10/11, python3.11, pytorch2.5.1, CUDA12.4

Notes, `pupil-apriltags` installation will be failed in python3.12. If you don't want to use our data capture method, you could skip this and choose python>=3.12.

## 2. Pretrained MaterialGAN model

The model weights will be automatically downloaded to the folder `ckp` when you run the scripts.


## 3.Quick try

`python run.py`

We provide a captured image set (`data/yellow_box-17.0-0.1/raw/*.jpg`), and corresponding JSON files. 
The generated results will be in the folder `data/yellow_box-17.0-0.1/optim_latent/1024/`, including generated SVBRDF maps (`nom.png `, `dif.png`, `spe.png`, `rgh.png`), re-rendered target images (`0*.png`) and relighting video in an environment map (`vid.gif`).

## 4. Usage

To optimize SVBRDF maps, we need several images with different lighting and a corresponding JSON file, which has all the information included.
If you use our dataset, all the JSON files are provided. If you want to capture new data, see below instruction. The JSON file will be generated automatically.

See `run.py` for more details.

# Real captured Dataset

## 1. Capture your own data with a smartphone

<img src="https://github.com/tflsguoyu/svbrdf-diff-renderer/blob/master/fig/fig1.png" width="600px">

<details>

<summary>Click to see more details</summary>

**Steps:**
1. Print "fig/tag36h11_print.png" on a solid paper with proper size and crop the center area.
2. Measure `size`(in cm unit) with a ruler, see the red arrow line in the below figure.
3. Place it on the material you want to capture and make the paper as flat as possible.
4. Turn on the camera flashlight and capture images from different views.
5. Create a data folder for captured images. We provide an example here, `data/yellow_box-17.0-0.1/raw`.
6. Run the script in `run.py`.
   ```bash
   gen_targets_from_capture(Path("data/yellow_box-17.0-0.1"), size=17.0, depth=0.1)
   ```
7. The generated target images are located in `data/yellow_box-17.0-0.1/target` and the corresponding JSON files are generated as well.

The `size` here is the number you measured from step 2; `depth` is the distance (in cm unit) between the marker plane and the material plane. For example, if you attach the markers on thick cardboard, you should use a larger `depth`.

**Tips:**
1. All markers should be captured and in focus and the letter `A` should be facing up.
2. It's better to capture during the night or in a dark room and turn off other lights.
3. It's better to see the highlights in the cropped area.
4. Change camera mode to manual, and keep the white balance and focal length the same during the captures.
5. `.heic` image format is not supported now. Convert it to `.png`/`.jpg` first.
6. Preferred capturing order: highlight in topleft -> top -> topright -> left -> center -> right -> bottomleft -> bottom -> bottomright. See images in `data/yellow_box/raw` as references.

</details>

## 2. The [[Dataset(38)](https://drive.google.com/file/d/1Vs2e35c4bNHRUu3ON4IsuOOP6uK8Ivji/view?usp=sharing)] used in this paper

The dataset includes corresponding JSON files. We put our results here as a reference, and you can also generate the results using our code from `run.py`.

```bash
optim_ganlatent(material_dir / "optim_latent_256.json", 256, 0.02, [1000, 10, 10], "auto")
optim_perpixel(material_dir / "optim_pixel_256_to_512.json", 512, 0.01, 20, tex_init="textures")
optim_perpixel(material_dir / "optim_pixel_512_to_1024.json", 1024, 0.01, 20, tex_init="textures")
```

For most of the cases, we use `auto` mode as the initialization. The results are shown below (input photos and output texture maps (1024x1024)),

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book1/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-blue/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-blue/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-red/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-red/optim_latent/1024/tex.jpg" width="128px">

<details>

<summary>Click to see more results</summary>

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag3/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag3/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-blue/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-blue/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-brown/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-brown/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-darkbrown/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-darkbrown/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-carpet/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-carpet/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-foam/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-foam/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-carton/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-carton/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-grid/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-grid/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/rubber-pattern/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/rubber-pattern/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bathroom-tile/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bathroom-tile/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bigtile/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bigtile/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-smalltile/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-smalltile/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-vinyl-floor/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-vinyl-floor/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-color/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-color/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-red-bump/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-red-bump/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-green/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-green/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-white/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-white/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-alder/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-alder/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-jatoba/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-jatoba/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-knotty/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-knotty/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-laminate/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-laminate/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-t/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-t/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-tile/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-tile/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-treeskin/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-treeskin/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-walnut/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-walnut/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-bamboo/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-bamboo/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bamboo-veawe/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bamboo-veawe/optim_latent/1024/tex.jpg" width="128px">

For some specular materials, the highlights are baked in the roughness maps. Using a lower roughness as initialization (`ckp = ["ckp/latent_const_W+_256.pt", "ckp/latent_const_N_256.pt"]`) will give better results,

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/optim_latent_spe/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/optim_latent_spe/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/optim_latent_spe/1024/tex.jpg" width="128px">

</details>

We also provide the code in `run.py` to generate novel-view renderings with an environment map,

```bash
render_envmap(material_dir / "optim_latent/1024", 256)
```

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile1/optim_latent/1024/vid.gif" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-blue/optim_latent/1024/vid.gif" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag1/optim_latent/1024/vid.gif" width="128px">

## 3. The [[Dataset_Zhou(76)](https://drive.google.com/file/d/1kfefC6YbkbSazLeJ7uUUUFR6WEeWozgA/view?usp=sharing)] from [Xilong Zhou](https://people.engr.tamu.edu/nimak/Papers/SIGAsia2022_LookAhead/index.html) with our JSON files.

The results are optimized by MaterialGAN,

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_blackleather1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_blackleather1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_blackleather2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_blackleather2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_laptop/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_laptop/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_leather/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_leather/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_metal/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_metal/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_redleather/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_redleather/optim_latent/1024/tex.jpg" width="128px">

<details>

<summary>Click to see more results</summary>

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_redmetal/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_redmetal/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_shinny_metal/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_shinny_metal/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_starfabric1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_starfabric1/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_starfabric2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_starfabric2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_wood1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_wood1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_wood2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/guo2_wood2/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_leather_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_leather_1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_plastic_2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_plastic_2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_stone_3/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_stone_3/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wall_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wall_1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wall_2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wall_2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_1/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_3/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_3/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_4/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_4/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_5/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_5/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_6/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_6/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_7/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_7/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_8/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_8/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_9/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_9/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_10/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home_wood_10/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_bluecanvas1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_bluecanvas1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_bluecanvas2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_bluecanvas2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_bluefabric/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_bluefabric/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_browncanvas/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_browncanvas/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_greyfabric/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_greyfabric/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_kitchen/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_kitchen/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_leather/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_leather/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_orangeleather/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_orangeleather/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_shoe1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_shoe1/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_shoe2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_shoe2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_table/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_table/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_yellowcanvas/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/home2_yellowcanvas/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/lab2_leather/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/lab2_leather/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/lab2_tile2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/lab2_tile2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_leather_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_leather_1/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_leather_2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_leather_2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_leather_3/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_leather_3/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_metal_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_metal_1/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_3/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_3/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_5/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_5/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_6/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_plastic_6/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_stone_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_stone_1/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_stone_2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_stone_2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_2/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_3/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_3/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_4/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_4/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_5/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wall_5/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_3/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_3/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_4/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_4/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_5/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_5/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_6/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_6/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_7/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_7/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_8/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_8/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_9/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima_wood_9/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_fabric1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_fabric1/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_fabric2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_fabric2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_metal/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_metal/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_plastic/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_plastic/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_shoe/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/nima2_shoe/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/peter_ground1/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/peter_ground1/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/peter_ground2/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/peter_ground2/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/peter_ground3/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/peter_ground3/optim_latent/1024/tex.jpg" width="128px">
&nbsp;&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/peter_metal/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/peter_metal/optim_latent/1024/tex.jpg" width="128px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/wen-laptop/target/all.jpg" width="128px">&nbsp;
<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data_xl/wen-laptop/optim_latent/1024/tex.jpg" width="128px">

</detail>
</details>

# Citation

If you find this work useful for your research, please cite:

```
@article{Guo:2020:MaterialGAN,
  title={MaterialGAN: Reflectance Capture using a Generative SVBRDF Model},
  author={Guo, Yu and Smith, Cameron and Ha\v{s}an, Milo\v{s} and Sunkavalli, Kalyan and Zhao, Shuang},
  journal={ACM Trans. Graph.},
  volume={39},
  number={6},
  year={2020},
  pages={254:1--254:13}
}
```

# Contacts

Welcome to report bugs and leave comments (Yu Guo: tflsguoyu@gmail.com)

