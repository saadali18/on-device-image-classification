# Knowledge Distillation for Efficient On-Device Image Classification

Deploying deep learning models on resource-constrained devices — mobile phones,
embedded sensors, edge nodes — requires models that are accurate, fast, and small.
Knowledge distillation offers an elegant solution: train a compact student network
to mimic the output distribution (soft labels) of a large pretrained teacher network, recovering much of the teacher’s accuracy at a fraction of the computational
cost. This project asks students to implement the Hinton et al. distillation objective (a combination of hard-label cross-entropy and KL divergence against softened
teacher logits), vary the temperature hyperparameter, and measure the studentteacher accuracy gap as a function of the student’s depth and width. Crucially,
students must conduct a systematic study of what the teacher is transferring: by
comparing student models trained on hard labels alone versus soft labels, they must
explain why soft probability distributions over incorrect classes (the “dark knowledge”) improve generalization. Advanced extensions include intermediate featurelevel distillation and comparing distillation against pruning and quantization as
complementary compression strategies. This project builds deep intuition for the
bias-variance-efficiency tradeoffs that are central to the course.

