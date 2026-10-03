# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
from dpdetector.distillation import binary_distillation_loss


class DistillationTests(unittest.TestCase):
    def test_equal_distributions_have_zero_kl(self):
        logits = torch.tensor([-12.,-1.,0.,3.,14.])
        self.assertLess(abs(binary_distillation_loss(logits,logits).item()),1e-6)

    def test_teacher_is_detached_and_student_gradient_has_correct_sign(self):
        student = torch.tensor([0.,0.],requires_grad=True)
        teacher = torch.tensor([8.,-8.],requires_grad=True)
        loss = binary_distillation_loss(student,teacher)
        loss.backward()
        self.assertLess(student.grad[0].item(),0)
        self.assertGreater(student.grad[1].item(),0)
        self.assertIsNone(teacher.grad)


if __name__ == '__main__':
    unittest.main()
