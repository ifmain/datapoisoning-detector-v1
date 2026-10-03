# SPDX-License-Identifier: Apache-2.0
import torch
from torch.nn import functional as F


def binary_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """Bernoulli KL distillation; does not require matching hidden dimensions."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    # Research basis: Hinton et al. (2015), arXiv:1503.02531; original binary KL implementation.
    student = torch.stack((torch.zeros_like(student_logits), student_logits.float()), dim=-1)/temperature
    teacher = torch.stack((torch.zeros_like(teacher_logits), teacher_logits.detach().float()), dim=-1)/temperature
    return F.kl_div(F.log_softmax(student,dim=-1),F.softmax(teacher,dim=-1),reduction='batchmean')*temperature**2
