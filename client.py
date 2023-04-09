import socket

import argparse, json, copy, os
import pickle

from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, RequestBasicsAgent, AgentDQN
from deep_dialog.usersims import RuleSimulator

from deep_dialog import dialog_config
from deep_dialog.dialog_config import *

from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg


class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def send_request(self, message):
        # create a socket object
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connect to the server
        client_socket.connect((self.host, self.port))
        # send a request to the server
        client_socket.send(message.encode())
        # receive a response from the server
        response = client_socket.recv(1024)
        print(response.decode())
        # close the client socket
        client_socket.close()

if __name__ == "__main__":
    client = Client("localhost", 8000)


    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_path", dest="dict_path", type=str, default="./deep_dialog/data/dicts.v3.p", help="path to the json dictionary file")
    parser.add_argument("--movie_kb_path", dest="movie_kb_path", type=str, default="./deep_dialog/data/movie_kb.1k.p", help="path to the movie kb .json file")
    parser.add_argument("--act_set", dest="act_set", type=str, default="./deep_dialog/data/dia_acts.txt", help="path to dia act set; none for loading from labeled file")
    parser.add_argument("--slot_set", dest="slot_set", type=str, default="./deep_dialog/data/slot_set.txt", help="path to slot set; none for loading from labeled file")
    parser.add_argument("--goal_file_path", dest="goal_file_path", type=str, default="./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p", help="a list of user goals")
    parser.add_argument("--diaact_nl_pairs", dest="diaact_nl_pairs", type=str, default="./deep_dialog/data/dia_act_nl_pairs.v6.json", help="path to the pre-defined dia_act&NL pairs")

    parser.add_argument("--max_turn", dest="max_turn", default=20, type=int, help="maximum length of each dialog (default=20, 0=no maximum length)")
    parser.add_argument("--episodes", dest="episodes", default=1, type=int, help="Total number of episodes to run (default=1)")
    parser.add_argument("--slot_err_prob", dest="slot_err_prob", default=0.05, type=float, help="the slot err probability")
    parser.add_argument("--slot_err_mode", dest="slot_err_mode", default=0, type=int, help="slot_err_mode: 0 for slot_val only; 1 for three errs")
    parser.add_argument("--intent_err_prob", dest="intent_err_prob", default=0.05, type=float, help="the intent err probability")
    
    parser.add_argument("--agt", dest="agt", default=9, type=int, help="Select an agent: 0 for a command line input, 1-6 for rule based agents")
    parser.add_argument("--usr", dest="usr", default=1, type=int, help="Select a user simulator. 0 is a Frozen user simulator.")
    
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=0, help="Epsilon to determine stochasticity of epsilon-greedy agent policies")
    
    # load NLG & NLU model
    parser.add_argument("--nlg_model_path", dest="nlg_model_path", type=str, default="./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p", help="path to model file")
    parser.add_argument("--nlu_model_path", dest="nlu_model_path", type=str, default="./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p", help="path to the NLU model file")
    
    parser.add_argument("--act_level", dest="act_level", type=int, default=0, help="0 for dia_act level; 1 for NL level")
    parser.add_argument("--run_mode", dest="run_mode", type=int, default=0, help="run_mode: 0 for default NL; 1 for dia_act; 2 for both")
    parser.add_argument("--auto_suggest", dest="auto_suggest", type=int, default=0, help="0 for no auto_suggest; 1 for auto_suggest")
    parser.add_argument("--cmd_input_mode", dest="cmd_input_mode", type=int, default=0, help="run_mode: 0 for NL; 1 for dia_act")
    
    # RL agent parameters
    parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=1000, help="the size for experience replay")
    parser.add_argument("--dqn_hidden_size", dest="dqn_hidden_size", type=int, default=60, help="the hidden size for DQN")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, help="gamma for DQN")
    parser.add_argument("--predict_mode", dest="predict_mode", type=bool, default=False, help="predict model for DQN")
    parser.add_argument("--simulation_epoch_size", dest="simulation_epoch_size", type=int, default=50, help="the size of validation set")
    parser.add_argument("--warm_start", dest="warm_start", type=int, default=1, help="0: no warm start; 1: warm start for training")
    parser.add_argument("--warm_start_epochs", dest="warm_start_epochs", type=int, default=100, help="the number of epochs for warm start")
    
    parser.add_argument("--trained_model_path", dest="trained_model_path", type=str, default=None, help="the path for trained model")
    parser.add_argument("-o", "--write_model_dir", dest="write_model_dir", type=str, default="./deep_dialog/checkpoints/", help="write model to disk") 
    parser.add_argument("--save_check_point", dest="save_check_point", type=int, default=10, help="number of epochs for saving model")
     
    parser.add_argument("--success_rate_threshold", dest="success_rate_threshold", type=float, default=0.3, help="the threshold for success rate")
    
    parser.add_argument("--split_fold", dest="split_fold", default=5, type=int, help="the number of folders to split the user goal")
    parser.add_argument("--learning_phase", dest="learning_phase", default="all", type=str, help="train/test/all; default is all")
    
   # print (parser)
    args = parser.parse_args()
    # print (args)
    params = vars(args)



    params_str = json.dumps(params, separators=(',', ':'))



   
    params_str2 = str(params_str)

    print(params_str2)

    client.send_request(params_str)

