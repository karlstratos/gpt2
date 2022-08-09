import argparse
import os


def main(args):
    import random
    import torch
    import torch.distributed as dist
    import transformers

    from copy import deepcopy
    from data import get_loaders
    from datetime import datetime
    from evaluation import compute_perplexity
    from file_handling import mkdir_optional
    from optimization import get_optim
    from transformers import set_seed, get_linear_schedule_with_warmup, \
        AutoConfig, AutoModelForCausalLM
    from util import Logger, check_distributed, strtime, count_parameters

    transformers.logging.set_verbosity_error()

    set_seed(args.seed)
    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1

    mkdir_optional(os.path.dirname(args.model))
    logger = Logger(log_path=args.model + '.log', on=is_main_process)
    logger.log(str(args))
    logger.log(f'rank {rank} local_rank {local_rank} world_size {world_size}',
               force=True)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group('nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.log(f'Using device: {str(device)}', force=True)

    loader_train, loader_val, loader_test = get_loaders(
        args.model_name, args.data_dir, args.max_length, args.seed,
        args.batch_size, args.no_shuffle, args.num_workers, is_distributed)

    num_training_steps = len(loader_train) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps) if \
        args.num_warmup_steps is None else args.num_warmup_steps

    config = AutoConfig.from_pretrained(args.model_name)

    # GPT2LMHeadModel
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    logger.log(f'{count_parameters(model)} parameters')

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        logger.log('Single-process single-device, no model wrapping')

    optimizer, _, step_scheduler = get_optim(
        model, args.optim, args.lr, args.momentum, args.weight_decay,
        'const', 'linear', num_warmup_steps, num_training_steps,
        args.epochs)
    logger.log(f'Training for {num_training_steps} steps ', newline=False)
    logger.log(f'({num_warmup_steps} warmup steps if used)')
    logger.log('-' * 80)

    # Training
    step = 0
    val_perp_best = float('inf')
    sd_best = None
    start_time = datetime.now()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.
        for batch in loader_train:
            X, M, I = [tensor.to(device) for tensor in batch]
            labels = X.masked_fill(~M, -100)
            output = model(X, attention_mask=M, labels=labels)
            loss = output.loss
            loss_sum += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            step_scheduler.step()
            model.zero_grad()
            step += 1

        loss_avg = loss_sum / len(loader_train)
        lr = optimizer.param_groups[0]['lr']
        log = f'Epoch {epoch:3d} | '
        log += f'step {step:5d} / {num_training_steps:5d} | '
        log += f'lr: {lr:.5f} | '
        log += f'loss: {loss_avg:10.5f} | '

        val_perp, _, num_preds = compute_perplexity(model, loader_val,
                                                          device)
        log += f'val perp: {val_perp:3.2f} ({num_preds} preds)'

        if val_perp < val_perp_best:
            val_perp_best = val_perp
            sd_best = deepcopy(model.state_dict())
            log += f' <-------------'

        logger.log(log)

    logger.log(f'\nDone training | total time {strtime(start_time)}')
    model.load_state_dict(sd_best)
    test_perp, _, num_preds = compute_perplexity(model, loader_test, device)
    logger.log(f'test perp: {test_perp:3.2f} ({num_preds} preds)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--model_name', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large',
                                 'gpt2-xl'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_warmup_steps', type=int, default=None)
    parser.add_argument('--warmup_ratio', type=float, default=.1)
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--clip', type=float, default=1.)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
