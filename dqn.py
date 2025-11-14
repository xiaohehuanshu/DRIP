import numpy as np
import random
import os, sys, time
import traceback
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from torch_geometric.data import Data

import multiprocessing
import multiprocessing.queues

from tensorboardX import SummaryWriter
import wandb

from logging_tool import LazyTee
from model import DQN_Agent_Transformer_GAT_PRE
from replay_buffer import ReplayBuffer, Transition, is_valid_transition
from env_fixed_state_length import LayoutGenerator
from utils.adjust_learning_rate import adjust_learning_rate
from utils.log_gather import gather_logs


class DQN:
    def __init__(self, config):
        self.config = config

        self.state_dim = LayoutGenerator.state_dim_val
        self.action_dim = LayoutGenerator.action_dim_val
        self.control_dim = LayoutGenerator.control_dim_val

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.online_net = DQN_Agent_Transformer_GAT_PRE(4, self.state_dim // 4, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=3e-4)
        self.target_net = DQN_Agent_Transformer_GAT_PRE(4, self.state_dim // 4, self.action_dim).to(self.device)
        self.update_target()

        # file paths for saving and loading model parameters
        self.save_path_para = config["para_path"]
        self.load_path_para = config["para_path"]

        # define hyperparameters
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.max_grad_norm = config["max_grad_norm"]
        self.batch_size = config["batch_size"]
        self.n_step = config["n_step"]

        # parallel processing setup
        mp.set_start_method("spawn")
        self.gpu_count = torch.cuda.device_count()
        self.worker_num = config["worker_num"]
        self.pool = Pool(self.worker_num)
        self.manager = mp.Manager()
        self.rollout_queue = self.manager.Queue()
        self.result_queue = {i: self.manager.Queue() for i in range(self.worker_num)}

        # create replay buffer
        self.dataset_size = 0
        self.training_step = 0
        self.buffer = ReplayBuffer(
            capacity=config["buffer_capacity"],
            alpha=config["buffer_alpha"],
            beta=config["buffer_beta"],
            beta_increment_per_sampling=config["buffer_beta_increment"],
            n_step=self.n_step,
            gamma=self.gamma,
        )

    def save(self, epoch):
        save_dir = f"./param/{self.save_path_para}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.online_net.state_dict(), f"{save_dir}/net_epoch{epoch}.pth")

    def load(self, epoch):
        path_para = self.load_path_para
        net_path = f"./param/{path_para}/net_epoch{epoch}.pth"
        self.online_net.load_state_dict(torch.load(net_path))
        self.update_target()

    def act(self, state, is_eval):
        graph_data = torch.tensor(state['control_edges'], dtype=torch.float).unsqueeze(0)

        state_tensor = {
            'state': torch.tensor(state['state'], dtype=torch.float32).unsqueeze(0).to(self.device),
            'mask': torch.tensor(state['mask'], dtype=torch.float32).unsqueeze(0).to(self.device),
            'controls_array': graph_data.to(self.device)
        }
        
        action_num = np.sum(state['mask'], dtype=np.int32)

        if is_eval:
            value_out = self.online_net(state_tensor)
            action = value_out.argmax().item()
        else:
            if random.random() < self.epsilon:
                value_out = torch.zeros((1, self.action_dim), dtype=torch.float32, device=self.device)
                action = random.randint(0, action_num - 1)
            else:
                value_out = self.online_net(state_tensor)
                action = value_out.argmax().item()

        if (action < 0) or (action > action_num - 1):
            print("Value of action =", action, "out of range=", action_num - 1, "!")
            print("Mask is ", state['mask'])
            print("Value of state is ", value_out)

        return action, value_out.detach().cpu().numpy().flatten()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update_target_soft(self, tau=0.08):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def update(self):
        if len(self.buffer) > self.batch_size:
            # get batch data
            batch_states, action, reward, batch_states_next, done, idxs, is_weight = self.buffer.sample(self.batch_size)

            # calculate Q values
            q_values = self.online_net(batch_states).gather(1, action)
            # target Q values
            with torch.no_grad():
                next_q_values = self.target_net(batch_states_next).max(1)[0].unsqueeze(1)
                target_q_values = reward + (1 - done) * self.gamma**self.n_step * next_q_values

            # update online network
            loss = (is_weight * F.mse_loss(q_values, target_q_values, reduction="none")).mean()
            if not torch.isfinite(loss):    # check for nan or inf
                print(f"Warning: Loss is {loss}")
                # check which part contains nan or inf
                batch_states_np = batch_states.cpu().detach().numpy()
                batch_states_next_np = batch_states_next.cpu().detach().numpy()
                has_nan_states = not np.all(np.isfinite(batch_states_np))
                has_nan_states_next = not np.all(np.isfinite(batch_states_next_np))
                # set numpy print options to show full arrays
                np.set_printoptions(threshold=np.inf, linewidth=2000, suppress=True)
                raise ValueError(
                    f"td_error invalid, containing nan or inf values\n"
                    f"q_values: {q_values}\n"
                    f"target_q_values: {target_q_values}\n"
                    f"batch_states invalid: {has_nan_states}\n"
                    f"batch_states_next invalid: {has_nan_states_next}\n"
                    f"batch_states: {batch_states_np}\n"
                    f"batch_states_next: {batch_states_next_np}\n"
                    f"action: {action}\n"
                    f"reward: {reward}\n"
                    f"done: {done}"
                )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if (self.training_step % self.dataset_size) == 0:
                self.writer.add_scalar("Loss", loss.item(), (int)(self.training_step / self.dataset_size))

            # update priorities after learning
            new_td_error = torch.abs(target_q_values - q_values).cpu().detach().numpy()
            self.buffer.update_priorities(idxs, new_td_error)

            self.training_step += 1
            # update target network every 20 steps
            # if self.training_step % 20 == 0:
            #     self.update_target()
            # update target network softly
            self.update_target_soft()

    def learn(self):
        init_time = time.time()

        mode = self.config["train_mode"]
        total_epoch = self.config["total_epoch"]

        best_reward = 0
        best_reward_list = []
        reward_total_list = []

        log_dir = self.config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        # configure log file
        log_filename = log_dir + "/output.log"
        sys.stdout = sys.stderr = LazyTee(log_filename)
        # initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
        # initialize wandb
        wandb.init(project=self.config["wandb_project"], config=self.config)

        if mode == "train":
            warmup_flag = True
            lr_max = self.config["lr_max"]
            lr_min = self.config["lr_min"]
            print("The trained model was initialized.")
        elif mode == "retrain":
            warmup_flag = False
            lr_max = self.config["lr_min"]
            lr_min = self.config["lr_min_retrain"]
            self.epsilon = self.epsilon_min
            self.load("_best_score")
            print("The pre-trained model was successfully loaded.")
        else:
            print("The mode is incorrect. The optional value is 'train' or 'retrain'.")
            return -1

        for i_epoch in range(total_epoch + 1):
            start_time = time.time()
            torch.cuda.empty_cache()
            reward_total, epoch_steps = self.rollout(is_eval=False, is_visible=False)
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            adjust_learning_rate(self.optimizer, i_epoch, 500, lr_min=lr_min, lr_max=lr_max, warmup=warmup_flag)
            epoch_time = time.time() - start_time

            if reward_total > best_reward:
                best_reward = reward_total
                self.save(epoch="_best_score")
                print(f"    Best model parameters updated.")

            reward_total_list.append(reward_total)
            best_reward_list.append(best_reward)
            # record training process
            self.writer.add_scalar("Reward", reward_total, i_epoch)
            self.writer.add_scalar("Best_Reward", best_reward, i_epoch)
            wandb.log({"Reward": reward_total, "Best_Reward": best_reward, "Epoch": i_epoch})
            print(
                "epoch",
                i_epoch,
                "time",
                f"{epoch_time:.2f}",
                "steps",
                epoch_steps,
                "reward",
                f"{reward_total:.2f}",
                "best",
                f"{best_reward:.2f}",
            )

            # save model parameters and evaluate every 20 epochs
            if (i_epoch % 20 == 0) and (i_epoch != 0):
            # if i_epoch % 20 == 0:
                self.save(epoch=i_epoch)
                print(f"    Model parameters saved.")
                start_time = time.time()
                reward_eval, epoch_steps_eval = self.rollout(is_eval=True, is_visible=False)
                self.writer.add_scalar("Eval_Reward", reward_eval, i_epoch)
                wandb.log({"Eval_Reward": reward_eval, "Epoch": i_epoch})
                epoch_time = time.time() - start_time
                print(
                    "eval epoch", i_epoch, "time", f"{epoch_time:.2f}", "steps", epoch_steps_eval, "reward", f"{reward_eval:.2f}"
                )

        training_time = time.time() - init_time
        print("Training is done! Time-consuming:", f"{training_time:.2f}")
        self.writer.close()
        wandb.finish()
        # restore stdout and stderr
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        # gather logs after training
        gather_logs(log_dir, output_filename="log_report.log")

    def predict(self):
        start_time = time.time()

        log_dir = self.config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        # configure log file
        log_filename = log_dir + "/output_predict.log"
        sys.stdout = sys.stderr = LazyTee(log_filename)
        
        # model_name_str = "1000"         # use specified epoch model parameters
        model_name_str = "_best_score"  # use best score model parameters
        self.load(model_name_str)
        print(f"Model {model_name_str} was successfully loaded.")
        reward_eval, epoch_steps_eval = self.rollout(is_eval=True, is_visible=True)
        epoch_time = time.time() - start_time
        print("eval epoch time", epoch_time, "steps", epoch_steps_eval, "reward", f"{reward_eval:.2f}")
        self.pool.close()
        self.pool.join()
        self.manager.shutdown()

        # restore stdout and stderr
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def rollout(self, is_eval=False, is_visible=False):
        sequential_reward = 0
        step_len = 0

        if is_eval:
            self.online_net.eval()
        else:
            self.online_net.train()

        if self.config["environment"] == "LayoutGenerator":
            # load all files from dataset path
            path_in = self.config["dataset_path"]
            log_dir = self.config["log_dir"]
            base_model_path = self.config["base_model_path"]
            all_files = []
            # traverse the dataset directory to get all files
            for root, dirs, files in os.walk(path_in):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), path_in)
                    all_files.append(relative_path)

            # sample files according to different modes
            if is_visible:
                sampled_files = all_files
                pbar = tqdm(total=len(sampled_files), file=sys.__stdout__)
            elif is_eval:
                sampled_files = random.sample(all_files, min(1500, len(all_files)))
                pbar = tqdm(total=len(sampled_files), file=sys.__stdout__)
            else:
                sampled_files = random.sample(all_files, min(500, len(all_files)))
            random.shuffle(sampled_files)

            results = []
            active_processes = 0
            for file_idx, file in enumerate(sampled_files):
                if file.endswith("xlsx") or file.endswith("csv"):
                    active_processes += 1
                    if self.gpu_count > 1:
                        # gpu_id = (active_processes-1) % (self.gpu_count-1)+1
                        gpu_id = active_processes % self.gpu_count
                    else:
                        gpu_id = 0
                    queue_id = file_idx % self.worker_num
                    result = self.pool.apply_async(
                        rollout_sub,
                        args=(
                            file,
                            path_in,
                            log_dir,
                            is_visible,
                            self.rollout_queue,
                            self.result_queue[queue_id],
                            queue_id,
                            base_model_path,
                            gpu_id,
                        ),
                    )
                    results.append(result)
            file_num = active_processes
            self.dataset_size = file_num
            while active_processes > 0:
                # reveive state from subprocesses, perform inference in main process, and send action back to subprocesses
                try:
                    queue_id, pid, state = self.rollout_queue.get(timeout=30)
                except multiprocessing.queues.Empty:
                    print("Timeout at rollout: self.rollout_queue.get()")
                    print("active_processes is", active_processes)
                    print(traceback.format_exc())
                    return
                if state is None:  # check for termination signal
                    active_processes -= 1
                    if not is_eval and not is_visible:
                        self.update()
                    else:
                        pbar.update(1)
                else:
                    action, value_out = self.act(state, is_eval)
                    self.result_queue[queue_id].put([pid, action, value_out])
            for result in results:
                sub_reward, sub_len, trans_buffer = result.get()
                sequential_reward += sub_reward
                step_len += sub_len
                if not is_eval and not is_visible:
                    if trans_buffer is not None:
                        for trans in trans_buffer:
                            self.buffer.add(trans)
                else:
                    pbar.close()
            sequential_reward /= file_num
        else:
            raise ValueError("Invalid environment type specified in CONFIG")

        return sequential_reward, step_len


def rollout_sub(file, path_in, log_dir, is_visible, rollout_queue, result_queue, queue_id, base_model_path, gpu_id):
    try:
        log_filename = log_dir + "/error_log/" + f"/{file}_output.log"
        sys.stdout = sys.stderr = LazyTee(log_filename)

        sub_reward = 0
        sub_len = 0
        trans_buffer = []
        unique_id = queue_id
        pid = os.getpid()

        worker = LayoutGenerator(base_model_path, gpu_id)
        state = worker.reset(path_in, file)

        message = None
        if state is None:
            message = f"All rooms in {file} have been placed."
        else:
            if np.sum(state['mask']) == 0:
                message = f"{file} floor{worker.floor_pointor} no initial feasible solution."
        if message is not None:
            print(message)
            rollout_queue.put((unique_id, pid, None), timeout=10)
            return 0, 0, None

        while True:
            try:
                rollout_queue.put((unique_id, pid, state), timeout=10)
            except multiprocessing.queues.Full:
                print("Timeout at rollout_sub: rollout_queue.put(state)")
                print(traceback.format_exc())
                return

            while True:
                try:
                    pid_get, action, value_out = result_queue.get(timeout=10)
                    if pid_get == pid:
                        break
                    else:
                        result_queue.put([pid_get, action, value_out])
                except multiprocessing.queues.Empty:
                    print(f"Warning: Timeout while waiting for action in process {unique_id}")
                    print(traceback.format_exc())
                    return

            next_state, reward, terminated, truncated, _ = worker.step(action, value_out, log_dir, is_visible)
            done = terminated or truncated

            if is_visible:
                worker.render()

            if is_valid_transition(state, next_state, action, reward):
                next_state_ = {
                    'state': np.zeros_like(state['state']),
                    'mask': np.zeros_like(state['mask']),
                    'control_edges': np.zeros_like(state['control_edges']),
                    'control_nodes': np.zeros_like(state['control_nodes'])
                } if done else next_state

                trans = Transition(state, next_state_, action, reward, done)
                trans_buffer.append(trans)

            sub_reward += reward
            sub_len += 1
            state = next_state
            if done:
                if worker.floor_pointor == worker.floor_count + 1:
                    break
                if np.sum(state['mask']) == 0:
                    message = f"{file} floor{worker.floor_pointor} no initial feasible solution."
                    print(message)
                    break

        # task finished, send termination signal None to main process
        try:
            rollout_queue.put((unique_id, pid, None), timeout=10)
        except multiprocessing.queues.Full:
            print("Timeout at rollout_sub: rollout_queue.put(None)")
            print(traceback.format_exc())
            return

    except Exception as e:
        print(f"An error occurred in rollout_sub {file}")
        try:
            worker.error_log(log_dir)
        except Exception as e:
            print(f"Error in calling worker.error_log: {e}")
        try:
            print("room_name is", worker.room_names[worker.room_pointer])
        except Exception as e:
            print(f"Error in printing room_name: {e}")
            print("floor_pointor is", worker.floor_pointor)
            print("length of room_names is", len(worker.room_names))
            print("room_pointer is", worker.room_pointer)
        try:
            print("Action is ", action)
        except Exception as e:
            print(f"Error in printing action: {e}")
            print("Implying reset failure")
            # no initial feasible solution, directly return
            print(traceback.format_exc())
            rollout_queue.put((unique_id, pid, None), timeout=10)
            return 0, 0, None
        try:
            print("Mask is ", state['mask'])
        except Exception as e:
            print(f"Error in calculating or printing mask: {e}")
        try:
            print("value_out is ", value_out)
        except Exception as e:
            print(f"Error in printing value_out: {e}")

        print(traceback.format_exc())
        rollout_queue.put((unique_id, pid, None), timeout=10)
        return 0, 0, None

    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return sub_reward, sub_len, trans_buffer
