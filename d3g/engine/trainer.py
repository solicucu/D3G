import datetime
import logging
import os
import time
import gc
import torch
import torch.distributed as dist

from d3g.utils.comm import get_world_size, synchronize
from d3g.utils.metric_logger import MetricLogger
from d3g.engine.inference import inference
from d3g.modeling.d3g import init_glance_memory


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    loss = loss.item()
    return loss

def is_larger(result1, result2, keys=["Avg_R1"]):
    if len(keys) == 1:
        return result1[keys[0]] > result2[keys[0]]
    elif len(keys) > 1:
        sum1 = sum(result1[key] for key in keys)
        sum2 = sum(result2[key] for key in keys)
        return sum1 > sum2
    else:
        raise RuntimeError("keys cannot be null!")
    
def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    param_dict,
    max_norm=5
):

    logger = logging.getLogger("d3g.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH

    model.train()
    start_training_time = time.time()
    end = time.time()
    max_iteration = len(data_loader)
    writer_count = 0
    best_result = {"Avg_R1":0, "Avg_R5":0}
    
    sample_seed = 0
    #### init the glance counter memory
    use_dga = cfg.SOLVER.USE_DGA
    num_clip = cfg.MODEL.D3G.NUM_CLIPS
    window = cfg.SOLVER.WINDOW 
    if use_dga:
        num_clip = cfg.MODEL.D3G.NUM_CLIPS
        window = cfg.SOLVER.WINDOW 
        data_loader.batch_sampler.sampler.set_epoch(sample_seed)
        memory = init_glance_memory(data_loader, num_clip, window)
        model.module.init_memory(memory)
    
    model.train()
    for epoch in range(arguments["epoch"], max_epoch + 1):
        rest_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch
        data_loader.batch_sampler.sampler.set_epoch(sample_seed)
        
        if epoch <= cfg.SOLVER.FREEZE_BERT:
            for param in param_dict['bert']:
                param.requires_grad_(False)
        else:
            for param in param_dict['bert']:
                param.requires_grad_(True)
        logger.info("Start epoch {}. base_lr={:.1e}, bert_lr={:.1e}, bert.requires_grad={}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], str(param_dict['bert'][0].requires_grad)))
        if epoch <= cfg.SOLVER.ONLY_IOU:
            logger.info("Using all losses")
        else:
            logger.info("Using only bce loss")
        
        for iteration, (batches, idx) in enumerate(data_loader):
            writer_count += 1
            iteration += 1
            batches = batches.to(device)
            optimizer.zero_grad()
            loss = model(batches, cur_epoch=epoch, idxs=idx)
            meters.update(loss=loss.detach()) 
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
          
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (max_iteration - iteration + rest_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 10 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            gc.collect()

        scheduler.step()
        
        # save last ckpt 
        checkpointer.save("model_last", **arguments)
        
        if (epoch % test_period == 0 and epoch >= cfg.SOLVER.SKIP_TEST):
            synchronize()
            torch.cuda.empty_cache()
            result_dict = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            # save the best result
            if dist.get_rank() == 0 and is_larger(result_dict, best_result, keys=["Avg_R1"]):
                best_result = result_dict 
                best_result["epoch"] = epoch
                logger.info("Got a new best result!")
                checkpointer.save("best_model", **arguments)
                
            synchronize()
            model.train()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
