level 7


distillation

    token-level knowledge distillation(minimize student's KL-div w/ teacher's prob)

        * access to teacher logits
        * relies on shared vocabulary/tokenizations
        * provides only local, step-wise supervision


    sentence-level:
    ...

    on-policy distillation:
        student generates its own trajectories & gets token-level feedback from teacher 

        backward-KL, mode-seeking
        

    on-policy self-distillation ?
        solves continual learning ?


 - KL-divergence web app

quantization
    trained weights tend to cluster near 0-1, but activations don't