from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup, get_constant_schedule


def get_optim(model, optim_type='sgd', lr=0.1, momentum=0, weight_decay=0,
              epoch_type='const', step_type='const', num_warmup_steps=0,
              num_training_steps=0, T_max=500):

    if optim_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                        momentum=momentum)
    elif optim_type == 'adam':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optim: ' + optim_type)

    # Using both epoch- and step-level scheduling seems to mess up lr updates.
    assert step_type == 'const' or epoch_type == 'const'

    if epoch_type == 'cos':
        # If T_max = num_epochs, no restart
        epoch_scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif epoch_type == 'const':
        epoch_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.)
    else:
        raise ValueError('Unknown epoch scheduler: ' + epoch_type)

    if step_type == 'const':
        step_scheduler = get_constant_schedule(optimizer)
    elif step_type == 'linear':
        step_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
    elif step_type == 'linconst':
        step_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps)
    else:
        raise ValueError('Unknown step scheduler: ' + step_type)

    return optimizer, epoch_scheduler, step_scheduler
