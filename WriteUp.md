# Object Detection in an Urban Environment

## Project Overview
Object detection is crucial for autonomous driving, detecting the objects surrounding the autonomous vehicle, allowing it to comprehend its location concerning other obstacles and avoiding collisions. This project creates a convolutional neural network to detect and classify objects using data from Waymo. From a dataset of images of urban environments containing annotated cyclists, pedestrians, and vehicles.
First, you will perform an extensive data analysis, including the computation of label distributions, display of sample images, and checking for object occlusions.

The project consists of 4 steps:
- step 1: Exploratory Data Analysis
- step 2: Edit the config file
- step 3: Model Training and Evaluation
- step 4: Improve the Performance

## Set up
The project has two options: using the classroom workspace, with the necessary libraries and data already available for you with cloud GPU, or a local setup. This project uses the classroom workspace, with 40 hours of GPU. Recommendation: while writing code, shut off your GPU, but to run the code, activate the GPU (some Python libraries will only be available with GPU on, such as TensorFlow).


To access the Jupyter Notebooks, use the environment web browser. Enter this code on the terminal:
```
cd /home/workspace
jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root
```

The environment by default comes with Firefox, however the version of Firefox in the environment is unstable with jupyter notebook so to make Firefox stable run the code below with no sandbox mode on the desktop environment.
```
sudo apt-get update
sudo apt-get install firefox
sudo firefox --no-sandbox
```

## Dataset
### Exploratory Data Analysis (EDA)
Data from Waymo is already present in ``/home/workspace/data/`` directory to explore the dataset.
- Implement the display_images function in the Exploratory Data Analysis notebook.
- From the dataset, some random images with bounding boxes for the class labels of vehicle, pedestrian, cycles.

| img 01                       | img 02                       | img 03                       | img 04                       | img 05                       |
|:----------------------------:|:----------------------------:|:----------------------------:|:----------------------------:|:----------------------------:|
| <img src="/img/img_01.png"/> | <img src="/img/img_02.png"/> | <img src="/img/img_03.png"/> | <img src="/img/img_04.png"/> | <img src="/img/img_05.png"/> |
| img 06                       | img 07                       | img 08                       | img 09                       | img 10                       |
| <img src="/img/img_06.png"/> | <img src="/img/img_07.png"/> | <img src="/img/img_08.png"/> | <img src="/img/img_09.png"/> | <img src="/img/img_10.png"/> |

- For additional EDA (see code on ``Exploratory Data Analysis.ipynb``), to comprehend the distribution between classes (vehicle, pedestrian, cycles) among the images:
<img src="/img/EDA_01.png"/>
<img src="/img/EDA_02.png"/>


## Cross-validation
A usual approach is to separate the dataset into 75% for Training, 15% for Validation and 10% for Testing since the Waymo open dataset used here has a good combination of a small number of classes in a wide variety of light and weather conditions. The Data of this project came split like this: Training with 86 images, Evaluation with 10 images, and Test with 3 images.


## Training
A **reference experiment** with a default "pipeline.config" with no extra data augmentation had a transfer learning using the SSD_ResNet50 model. Then, to improve the performance, one way is to improve the reference experiment by editing the ``pipeline_new.config`` in its experiment folder, then training for a new outcome:

### **Reference Experiment**: 
Config file on folder ```./experiments/reference```

- Momentum Optimizer:  ``cosine_decay_learning_rate``
- Data Augmentation Options: No extra augmentation

<img src="/img/tensorboard_01.png"/>
<img src="/img/tensorboard_02.png"/>


### **Experiment 0**: 
config file on folder ```./experiments/experiment0```

- Momentum Optimizer:  ``exponential_decay_learning_rate``
- Data Augmentation Options: No extra augmentation

<img src="/img/tensorboard_experiment0_01.png"/>
<img src="/img/tensorboard_experiment0_02.png"/>


**Analysis**: experiment0 vs reference experiment

Overall, the training results from experiment0, in comparison with the reference experiment, show that the model has improved a lot. 
Both models run for a little over 2.4k epochs until the learning rate decays to zero. But if compared, Total Loss experiment0 has reached 0.7, while the reference experiment kept at 10, proving that the Momentum Optimizer exponential_decay_learning_rate has outperformed cosine_decay_learning_rate.


### **Experiment 1**: 
Config file on folder ```./experiments/experiment1```

- Momentum Optimizer:  ``cosine_decay_learning_rate``
- Data Augmentation Options:
    1. Random RGB to GRAYSCALE conversion with a probability of 30%.
    2. Random Brightness Adjustment of Max difference of 30%.

<img src="/img/tensorboard_experiment1_01.png"/>
<img src="/img/tensorboard_experiment1_02.png"/>


**Analysis**: experiment1 vs reference experiment

Overall, the training results from experiment1, compared to the reference experiment, show that both models had similar performance. Both models run for a little over 2.4k epochs until the learning rate decays to zero. But if compared, Total Loss experiment1 reached 10 at a faster pace at 400 epochs and stabilized, while the reference experiment Total Loss at 10 and kept decreasing for more extended epochs. Grayscale and brightness augmentation deteriorate performance.

| experiment1: Image Augmented 01           | experiment1: Image Augmented 02           |
|:-----------------------------------------:|:-----------------------------------------:|
| <img src="/img/img_augment_exp1_01.png"/> | <img src="/img/img_augment_exp1_02.png"/> |


### **Experiment 2**: 
Config file on folder ```./experiments/experiment2``` 

- Momentum Optimizer:  ``exponential_decay_learning_rate``
- Data Augmentation Options:
    1. Random RGB to GRAYSCALE conversion with a probability of 30%.
    2. Random Brightness Adjustment of Max difference of 30%.
    3. Random Black Square Patches, max 10 per image, with a probability of 50% and a Size to Image Ratio of 5%.

<img src="/img/tensorboard_experiment2_01.png"/>
<img src="/img/tensorboard_experiment2_02.png"/>


**Analysis**: experiment2 vs reference experiment

Overall, the training results from experiment2, compared with the reference experiment, show that the model has improved a lot, gathering the benefits of the Momentum Optimizer exponential_decay_learning_rate even with three augmentation methods: grayscale, brightness, and black square occlusions. Both models run for a little over 2.4k epochs until the learning rate decays to zero. But Total Loss experiment2 has reached 1, improving its performance and bringing a greater diversity, becoming a more robust a versatile mode, as seen in the images below.

| experiment2: Image Augmented 01           | experiment2: Image Augmented 02           |
|:-----------------------------------------:|:-----------------------------------------:|
| <img src="/img/img_augment_exp2_01.png"/> | <img src="/img/img_augment_exp2_02.png"/> |

