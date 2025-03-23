
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(student_logits, teacher_logits, T):
    student_log_softmax = F.log_softmax(student_logits/T, dim=1)
    teacher_softmax = F.softmax(teacher_logits/T, dim=1)
    ce = -(teacher_softmax * student_log_softmax).sum(dim=1)
    return ce.mean()