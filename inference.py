import os, sys, time
import traceback
import random
import torch
import numpy as np
from tqdm import tqdm
from env_fixed_state_length import LayoutGenerator
from model import DQN_Agent_Transformer_GAT_PRE
from logging_tool import LazyTee
from torch_geometric.data import Data



class InferenceEngine:
    def __init__(self, model_path, base_model_path):
        self.state_dim = LayoutGenerator.state_dim_val
        self.action_dim = LayoutGenerator.action_dim_val
        self.control_dim = 33

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.online_net = DQN_Agent_Transformer_GAT_PRE(4, self.state_dim // 4, self.action_dim).to(self.device)
        self.online_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.online_net.eval()

    def act(self, state, is_eval):
        graph_data = torch.tensor(state['control_edges'], dtype=torch.float32)
        state_tensor = {
            'state': torch.tensor(state['state'], dtype=torch.float32).unsqueeze(0).to(self.device),
            'mask': torch.tensor(state['mask'], dtype=torch.float32).unsqueeze(0).to(self.device),
            'controls_array': graph_data.unsqueeze(0).to(self.device)
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

    def infer(self, path_in, file, log_dir, path_graph=False, is_visible=True):
        try:
            log_filename = log_dir + "/error_log/" + f"/{file}_output.log"
            sys.stdout = sys.stderr = LazyTee(log_filename)

            sub_reward = 0
            sub_len = 0

            worker = LayoutGenerator(base_model_path, gpu_id=0)
            state = worker.reset(path_in, file, path_graph)

            message = None
            if state is None:
                message = f"All rooms in {file} have been placed."
            else:
                if np.sum(state['mask']) == 0:
                    message = f"{file} floor{worker.floor_pointor} no initial feasible solution."
            if message is not None:
                print(message)
                return

            while True:
                action, value_out = self.act(state, is_eval=True)
                next_state, reward, terminated, truncated, _ = worker.step(action, value_out, log_dir, is_visible)
                done = terminated or truncated

                if is_visible:
                    worker.render()

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
                print(traceback.format_exc())
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

        finally:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        return


if __name__ == "__main__":
    start_time = time.time()
    # example usage
    model_path = "./param/net_param/net_epoch_best_score.pth"  # rl model path
    base_model_path = "./base_model"  # base model path

    # init inference engine
    engine = InferenceEngine(model_path, base_model_path)

    # log directory
    log_dir = "./logs/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))

    path_in = './dataset/data_eval/'  # input data path
    all_files = []
    for root, dirs, files in os.walk(path_in):
        for idx, file in enumerate(files):
            if (file.endswith("xlsx") or file.endswith("csv")) and ('graph' not in file):
                relative_path = os.path.relpath(os.path.join(root, file), path_in)
                all_files.append(relative_path)

    pbar = tqdm(total=len(all_files))
    for file_idx, file in enumerate(all_files):
        try:
            if file.endswith('csv'):
                file_name = file[:-4]
            else:
                file_name = file[:-5]

            if os.path.exists(path_in + file_name + '_graph.xlsx'):
                engine.infer(path_in, file, log_dir, path_graph=path_in + file_name + '_graph.xlsx')
            else:
                engine.infer(path_in, file, log_dir, path_graph=False)
        except:
            pass
        pbar.update(1)
    pbar.close()

    print(f"finished, time {time.time() - start_time:.2f} s")