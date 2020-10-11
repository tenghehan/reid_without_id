## 核心脚本使用

### yolov3_deepsort_ims.py

功能：使用deepsort算法对图片序列形式的视频作tracking。

运行：

```
--config_file (声明模型结构的yaml文件路径，暂时使用'fastreid_configs/MOT/bagtricks_R50.yml'(ResNet50 + BNNeck))

IMAGES_PATH (图片序列的路径，例'image_sequence/MOT16-05')

--fps (帧率，目前使用的视频fps都已知，采用原视频的fps可以保证生成的视频和原视频同速)

--model_path (deepsort中使用的reid model的weights，如果未提供model_path，则使用ImageNet pretrained BNNeck model)
```
输出：

    output/MOT16-05.avi
    output/MOT16-05.txt




### track2reid.py
功能：将tracking的结果转化为能用来训练reid模型的reid数据集。

运行：
 ```
--dataset_name (例MOT16-05)

--sampling_rate (采样率，图片序列中的每张图有sampling_rate的概率被采用，为了缓解相邻图片太过相似的问题，目前sampling_rate为默认值1)

--partition_rate (training数据集中id的数量占总id数量的比例，目前partition_rate为默认值0.8)

```
输出：
```
   reid_dataset/MOT16-05
       train/
            000001_c1_000320.jpg
            000001_c1_000321.jpg
            ...
       test/
            000002_c1_000129.jpg
            000002_c1_000130.jpg
            ...
       query/
            000002_c2_000019.jpg

      info.txt (生成的数据集的详细信息，包括id数量、image数量等)

图片名称: id_camid_frameindex/jpg

note: 在我们生成的数据集中实际上只有一个camera，但是为了数据集能够用fast_reid库进行训练，在图片命名时将test和query分别设置为camid=1和2。
```

### train_net.py
功能：用指定数据集训练reid模型。

运行：
```
--config-file (声明模型结构的yaml文件路径，暂时使用'fastreid_configs/MOT/bagtricks_R50.yml'(ResNet50 + BNNeck))

--specific_dataset (例MOT16-05)

--finetune
MODEL.WEIGHTS (以此路径下的model weights为基础继续训练模型)
```

输出：

```
logs/mot/bagtricks_R50/MOT16-05
    config.yaml
    log.txt
    metrics.json
    model_final.pth
```

功能：测试reid模型的准确度(DukeMTMC Market1501)

运行：
```
--eval_only
--config-file (声明模型结构的yaml文件路径，暂时使用'fastreid_configs/MOT/bagtricks_R50.yml'(ResNet50 + BNNeck))
DATASETS.TESTS ("DukeMTMC",) or ("Market1501",) 
MODEL.WEIGHTS logs/mot/bagtricks_R50/MOT16-06/model_final.pth
OUTPUT_DIR "logs/mot/bagtricks_R50/MOT1-16-06/dukemtmc or market1501"
```

输出：

```
logs/mot/bagtricks_R50/MOT16-05/dukemtmc or market1501
    config.yaml
    log.txt
```

加速：

```
https://github.com/JDAI-CV/fast-reid/blob/master/docs/GETTING_STARTED.md#compile-with-cython-to-accelerate-evalution
```

### eval_motchallenge.py

MOTChallenge 官方的 evaluation 脚本：https://github.com/dendorferpatrick/MOTChallengeEvalKit/blob/master/MOT/README.md

上述 README 中提到的另一个 MOT metrics 计算的库，指标计算与官方没有差异：https://github.com/cheind/py-motmetrics

在虚拟环境中 conda install numpy scipy pandas，pip install motmetrics 后，
```
python -m motmetrics.apps.eval_motchallenge image_sequence/ output/
```
指标主要关注最前面三个

```
image_sequence/
   MOT16-05/
     gt/gt.txt
   MOT16-12/
     gt/gt.txt
   ...

output/
   MOT16-05.txt
   MOT16-12.txt
   ...

文件结构需要满足上述格式，脚本会自动根据名字对对应脚本作evaluate。
```

## 自动化脚本使用

### track and train
#### 思路
```
    easy to hard: dataset01, dataset02,...
    ImageNet pretrained reid model -> model_0
    
    step 1: 使用model_0对dataset01作tracking，得到的结果转化为reid dataset: reid_dataset_01.
    step 2: 使用reid_dataset_01训练reid model，得到model_1.

    step3: 使用model_1对dataset02作tracking，得到的结果转化为reid dataset：reid_dataset_02.
    step4: 使用reid_dataset_02训练reid model，得到model_2.

    ...
```
#### 使用
##### 1.configs/auto.yaml
```
model_config: fastreid_configs/MOT/bagtricks_R50.yml
datasets:
    - name: campus4-c0
      fps: 25
      sampling_rate: 0.2
    - name: terrace1-c0
      fps: 25
      sampling_rate: 0.2
    - name: passageway1-c0
      fps: 25
      sampling_rate: 0.2
    - name: MOT16-05
      fps: 14
      sampling_rate: 0.5
    ...
```
##### 2.auto_train.py
```
python auto_train.py --config_file configs/auto.yaml --dry-run
只输出即将被顺序执行的若干命令

python auto_train.py --config_file configs/auto.yaml
按照yaml中的数据集开始完整的流程
```

### reid evaluate

#### 思路
```
    将训练得到的reid model作批量化的evaluation，包括DukeMTMC和Market1501
```

#### 使用
##### 1.configs/auto.yaml
```
model_config: fastreid_configs/MOT/bagtricks_R50.yml
datasets:
    - name: campus4-c0
      fps: 25
      sampling_rate: 0.2
    - name: terrace1-c0
      fps: 25
      sampling_rate: 0.2
    - name: passageway1-c0
      fps: 25
      sampling_rate: 0.2
    - name: MOT16-05
      fps: 14
      sampling_rate: 0.5
    ...
```
##### 2.auto_eval_reid.py
```
python auto_eval_reid.py --config_file configs/auto.yaml --dry-run
只输出即将被顺序执行的若干命令

python auto_eval_reid.py --config_file configs/auto.yaml
按照yaml中的数据集开始完整的流程
```

### auto tracking

#### 思路
```
    令model_n对dataset_n作tracking。
    其中model_n是利用(reid_dataset_01, reid_dataset_02..., reid_dataset_(n-1))训练得到的模型。
```

#### 使用
##### 1.configs/auto.yaml
```
model_config: fastreid_configs/MOT/bagtricks_R50.yml
datasets:
    - name: campus4-c0
      fps: 25
      sampling_rate: 0.2
    - name: terrace1-c0
      fps: 25
      sampling_rate: 0.2
    - name: passageway1-c0
      fps: 25
      sampling_rate: 0.2
    - name: MOT16-05
      fps: 14
      sampling_rate: 0.5
    ...
```
##### 2.auto_track.py
```
python auto_track.py --config_file configs/auto.yaml --dry-run
只输出即将被顺序执行的若干命令

python auto_track.py --config_file configs/auto.yaml
按照yaml中的数据集开始完整的流程
```

