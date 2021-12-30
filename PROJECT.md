## Project overview 
#### Object Detection in Urban Environment
This project is focused on utilizing machine learning to identify objects in the environment. It is crucial part of every autonomous system, as the vehicle needs to be aware of its surrounding and act according to it. One of the examples can be path planning, which should utilize detected objects class and localization to find the best way around.

## Set up
Most of the code was run on the local machine, using provided Dockerfile. \
Data was downloaded from Udacity's workspace, to avoid processing on local machine. \
Training and evaluating the model was done on Udacity's VM, due to lack of appropriate GPU. \

To clone the repository use: 
```
git clone https://github.com/nikodemj9/nd013-c1-vision-starter.git
```



python ./create_splits.py --source ./data/waymo/training_and_validation/ --destination ./data/waymo

python edit_config.py --train_dir /home/workspace/data/waymo/train/ --eval_dir /home/workspace/data/waymo/val/ --batch_size 4 --checkpoint /home/workspace/experiments/pretrained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt

mv pipeline_new.config experiments/reference/

cd experiments/

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config





This section should contain a brief description of the steps to follow to run the code for this repository.

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
Image augmentation added:
- 


This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
