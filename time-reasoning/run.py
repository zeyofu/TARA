import json
import os
import argparse
import nltk
nltk.download('punkt')
import torch
from finetune_and_evaluate import finetune_full, zero_shot_evaluation_nyt, quick_clip_call_for_attention, count_labelled_objects



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default=2, type=int, help='1:time, 2:geo')
    parser.add_argument('--to', default=2, type=int, help='1: Simple sentence, 2: Natural sentence')
    parser.add_argument('--name', default='default', help='Name of experiment')
    parser.add_argument('--mode', default='train', help='Choose betweeen train or test')
    parser.add_argument('--epochs', type=int, default=1, help='Choose number of epochs to train for')
    parser.add_argument('--model_path', type=str, default='none', help='Required if test mode is on')
    parser.add_argument('--lr', type=float, default=0.000001, help='Learning rate module')
    parser.add_argument('--train_in_eval', type=int, default=0, help='Clip sometimes should be finetuned with norm layers frozen. Idk CLIP is weird.')
    parser.add_argument('--num_negatives', type=int, default=19, help='Number of negatives')
    parser.add_argument('--wandb_proj', type=str, default='time-reasoning', help='Project name on wandb')
    parser.add_argument('--wandb_entity', type=str, default='ipc2107', help='Run entity on wandb')
    args = parser.parse_args()

    return args

def finetune(args):

    if args.name == '0':
        args.name = None


    if args.mode=='train':


        finetune_full(experiment_type=args.type, template_options=args.to, jsonl_path='/shared/xingyu/projects/visual/data/nyt_preprocessed/input', mode='train', epochs=int(args.epochs), args=args)


    elif args.mode=='test':

        if args.model_path == 'none' or not os.path.exists(args.model_path):
            raise Exception("Test Mode is being used without a provided model path or with a non-existent model path")

        finetune_full(experiment_type=args.type, template_options=args.to, jsonl_path='/shared/xingyu/projects/visual/data/nyt_preprocessed/input/', mode='test', model_path=args.model_path, args=args)

    else:
        raise Exception("Invalid mode. Choose between train and test")


def only_evaluate():
    args = get_args()

    if args.name == '0':
        args.name = None

    args.cl = int(args.cl)
    recall1, recall3, recall5 = zero_shot_evaluation_nyt(int(args.type), int(args.to), name=args.name)
    print("Recall@1: ")
    print(recall1)
    print("Recall@3: ")
    print(recall3)
    print("Recall@5: ")
    print(recall5)


def count_objects():
    all_object_counts = count_labelled_objects()
    with open('o_counts.json', 'w') as f:
        json.dump(all_object_counts, f)

def get_clip_dataset_self_attention(provided=0):

    quick_clip_call_for_attention(provided_start=provided)


def relevant_config():

    sweep_config = {

    }

    pass

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    args = get_args()
    if args.name == 'default':
        raise Exception("Please name the experiment")

    finetune(args)
