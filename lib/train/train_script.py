import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.stark import build_starks, build_starkst, build_simtrack
from lib.models.stark import build_stark_lightning_x_trt
# forward propagation related
from lib.train.actors import STARKSActor, STARKSTActor, SimTrackActor
from lib.train.actors import STARKLightningXtrtActor
# for import modules
import importlib

#将之前得到的参数根据不同模型进行训练
def run(settings): 
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):   #如果settings.cfg_file不存在则继续运行
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name) #将选择模型的config返回
    cfg = config_module.cfg #将该模型的cfg返回
    config_module.update_config_from_file(settings.cfg_file) #该函数位于其对应模型的config文件内，作用为更新config
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key]) #按顺序输出cfg内每个属性及对应的内容
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg) 

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs') 
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir) #递归创建多层目录
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name)) #在simtrack-config目录下建立log文件

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "stark_s":
        net = build_starks(cfg)
    elif settings.script_name == "stark_st1" or settings.script_name == "stark_st2":
        net = build_starkst(cfg)
    elif settings.script_name == "stark_lightning_X_trt":
        net = build_stark_lightning_x_trt(cfg, phase="train")
    elif settings.script_name == "simtrack":
        net = build_simtrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0") #选择设备gpu0
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False) #将config文件中对应值返回
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL") #KL?
    # Loss functions and Actors
    if settings.script_name == "stark_s" or settings.script_name == "stark_st1":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "stark_st2":
        objective = {'cls': BCEWithLogitsLoss()}
        loss_weight = {'cls': 1.0}
        actor = STARKSTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "stark_lightning_X_trt":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT} 
        actor = STARKLightningXtrtActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "simtrack":
        objective = {'giou': giou_loss, 'l1': l1_loss}  #计算此时的giou和L1-LOSS
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = SimTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings) #进行训练
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg) #？从net和cfg中获得optimizer, lr_scheduler（优化器和学习测略）
    use_amp = getattr(cfg.TRAIN, "AMP", False) #SIMTRACK无此参数，其他模型有
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    if settings.script_name in ["stark_st2", "stark_st2_plus_sp"]:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, load_previous_ckpt=True)
    else:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
