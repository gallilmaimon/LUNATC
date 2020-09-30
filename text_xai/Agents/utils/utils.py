from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.  # terminal state
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, ep_r, name):
    with global_ep.get_lock():
        global_ep.value += 1

    print(name, "Ep:", global_ep.value, "| Ep_r: %.5f" % ep_r)


def synced_update(opt, lnet, gnet, done, s_, bs, ba, br, gamma, sync_barrier, sync_event, s_queue, v_target_queue, a_queue):
    # calculate target for loss
    if done:
        v_s_ = 0.  # terminal state
    else:
        v_s_ = gnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    # collect experiences from all workers
    a_queue.put(ba)
    v_target_queue.put(buffer_v_target)
    s_queue.put(bs)

    # with one worker only calculate total loss and update parameters
    sync_event.clear()  # makes sure the event which makes sure calculation happened is false before starting
    id = sync_barrier.wait(timeout=60)
    if id == 0:
        ba = []
        bs = []
        buffer_v_target = []
        for _ in range(sync_barrier.parties):
            ba += a_queue.get()
            bs += s_queue.get()
            buffer_v_target += v_target_queue.get()

        opt.zero_grad()
        loss = gnet.loss_func(
            v_wrap(np.vstack(bs)),
            v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
            v_wrap(np.array(buffer_v_target)[:, None]))

        loss.backward()

        opt.step()
        sync_event.set()

    # update all local networks after finished updating global network
    sync_event.wait()
    lnet.load_state_dict(gnet.state_dict())
