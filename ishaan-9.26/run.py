import json
import os
import argparse
import nltk
nltk.download('punkt')
import torch
torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_start_method('spawn')
from datapoints import TimePoint, Location
from finetune_and_evaluate import finetune_full, zero_shot_evaluation_nyt, quick_clip_call_for_attention, count_labelled_objects
import wandb



def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--i_s', required=True, type=int, help='Start index of given input to be considered')
    #parser.add_argument('--i_e', required=True, type=int, help='End index + 1 of given input to be considered')
    parser.add_argument('--type', default=2, type=int, help='1:time, 2:geo')
    parser.add_argument('--to', default=2, type=int, help='1: Simple sentence, 2: Natural sentence')
    parser.add_argument('--cl', default='1', help='List the number of levels to count as correct')
    parser.add_argument('--name', default='debug', help='Name of experiment')
    parser.add_argument('--mode', default='train', help='Choose betweeen train or test')
    parser.add_argument('--epochs', type=int, default=1, help='Choose number of epochs to train for')
    parser.add_argument('--model_path', type=str, default='none', help='Required if test mode is on')
    args = parser.parse_args()

    return args

def finetune():

    wandb.config.direction = 'finetune'
    args = get_args()
    wandb.config.update(args)

    if args.name == '0':
        args.name = None


    if args.mode=='train':


        finetune_full(experiment_type=args.type, template_options=args.to, correctness_level=int(args.cl), jsonl_path='input_zip_nyt_dataset/details', mode='train', epochs=int(args.epochs))


    elif args.mode=='test':

        if args.model_path == 'none' or not os.path.exists(args.model_path):
            raise Exception("Test Mode is being used without a provided model path or with a non-existent model path")

        finetune_full(experiment_type=args.type, template_options=args.to, correctness_level=int(args.cl), jsonl_path='input_zip_nyt_dataset/details', mode='test', model_path=args.model_path)

    else:
        raise Exception("Invalid mode. Choose between train and test")


def only_evaluate():
    args = get_args()

    if args.name == '0':
        args.name = None

    args.cl = int(args.cl)
    recall1, recall5, recall10 = zero_shot_evaluation_nyt(int(args.type), int(args.to), correctness_level=int(args.cl), name=args.name)
    print("Recall@1: ")
    print(recall1)
    print("Recall@5: ")
    print(recall5)
    print("Recall@10: ")
    print(recall10)


def count_objects():
    all_object_counts = count_labelled_objects()
    with open('o_counts.json', 'w') as f:
        json.dump(all_object_counts, f)

def get_clip_dataset_self_attention(provided=0):

    quick_clip_call_for_attention(provided_start=provided)


if __name__ == "__main__":


    wandb.login()
    wandb.init(project='time-reasoning', entity='ipc2107')
    wandb.run.name = 'debug'
    wandb.run.save()

    finetune()
    wandb.finish()