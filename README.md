# Knowledge Distillation (KD) with Pytorch

## Usage

```bash
$ git clone git@github.com:HHorimoto/pytorch-kd.git
$ cd pytorch-kd
$ ~/python3.10/bin/python3 -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
$ source run_teacher.sh # large model
$ source run_student.sh # light model
```

## Features

### Knowledge Distillation
I trained a large model (Teacher), a light model (Student) and a light model with knowledge distillation (Student-kd) for 50 epochs using CIFAR-10.
The table below presents the results obtained from the experiments.

**Comparison Table**

The table shows that **Student-kd** achieves higher accuracy than **Student**, demonstrating the benefit of knowledge distillation. While the Teacher model has the best accuracy, it is much larger and slower. **Student-kd** retains the efficiency of Student but with improved performance, though at the cost of slightly longer training time.

|            |  Accuracy  | Elapsed Time |  Parameters   |
| ---------- | :--------: | :----------: | :-----------: |
| Teacher    | **0.7836** |   516.643    |  12,626,890   |
| Student    |   0.7546   | **455.307**  | **3,163,114** |
| Student-kd |   0.7632   |   514.012    | **3,163,114** |


#### Reference
[1] [https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/10_knowledge_distillation.ipynb](https://github.com/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/11_cnn_pytorch/10_knowledge_distillation.ipynb)