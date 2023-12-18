"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults, #模型和扩散的默认参数（是一个字典）
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist() #创建一个分布式进程组
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())#根据默认的参数key获取命令行参数
    ) #根据命令行参数，和默认参数，创建模型和扩散
    model.to(dist_util.dev()) #将模型放到指定的设备上
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)#根据命令行参数创建一个采样调度器（有两种：uniform和loss-second-moment）

    logger.log("creating data loader...")
    data = load_data( #加载数据，返回一个生成器，生成（images, kwargs）对，images是一个NCHW浮点张量，kwargs字典包含零或多个键，如果有类别标签，键是“y”，值是整数张量的类标签
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values EMA值的逗号分隔列表
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())#使用guided_diffusion.script_util里的默认值更新这里的默认值
    parser = argparse.ArgumentParser() #创建一个解析器
    add_dict_to_argparser(parser, defaults) #将默认值（defaults是一个字典）添加到解析器
    return parser


if __name__ == "__main__":
    main()
