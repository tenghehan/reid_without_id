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
--config_file (声明模型结构的yaml文件路径，暂时使用'fastreid_configs/MOT/bagtricks_R50.yml'(ResNet50 + BNNeck))

--specific_dataset (例MOT16-05)

--finetune
--model_path (以此路径下的model weights为基础继续训练模型)
```

输出：

```
logs/mot/bagtricks_R50/MOT16-05
    config.yaml
    log.txt
    metrics.json
    model_final.pth
```

## 自动化脚本使用

### 思路
```
    easy to hard: dataset01, dataset02,...
    ImageNet pretrained reid model -> model_0
    
    step 1: 使用model_0对dataset01作tracking，得到的结果转化为reid dataset: reid_dataset_01.
    step 2: 使用reid_dataset_01训练reid model，得到model_1.

    step3: 使用model_1对dataset02作tracking，得到的结果转化为reid dataset：reid_dataset_02.
    step4: 使用reid_dataset_02训练reid model，得到model_2.

    ...
```
### 使用
#### 1.configs/auto.yaml
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
    - name: MOT16-12
      fps: 30
      sampling_rate: 0.2
    - name: MOT16-06
      fps: 14
      sampling_rate: 0.5
```
#### 2.auto.py
```
python auto.py --config_file configs/auto.yaml --dry-run
只输出即将被顺序执行的若干命令

python auto.py --config_file configs/auto.yaml
按照yaml中的数据集开始完整的流程
```