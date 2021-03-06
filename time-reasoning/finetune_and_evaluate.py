import json
import os
import random
from PIL import Image
import argparse
from utils import custom_collate, get_indices_given_split
from torch.optim import SGD, Adam
import re
import nltk
import cv2
nltk.download('punkt')
import numpy as np
import math
import torch
from torch.utils import data
from tqdm import tqdm
import clip
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F_nn
from datapoints import Location, TimePoint
from nyt_dataset import GeoTemporalDataset, recover_time_params
from nyt_image import GeoTemporalImage
from torch.profiler import profile, record_function, ProfilerAction, ProfilerActivity


def pil_loader(path):

    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    except:
        print(f'Error with the image at {path}')
        return Image.new('RGB', (300, 300))

def get_coco_categories():

    json_file ='../clip_guided_generation/annotations/instances_val2017.json'
    cat_dict = {}
    if json_file is not None:
        with open(json_file,'r') as COCO:
            js = json.loads(COCO.read())
            for each in js['categories']:
                cat_dict[int(each["id"])] = each["name"]

    return cat_dict

def get_transform():

    transforms = []
    transforms.append(T.ToTensor())

    return T.Compose(transforms)

def get_object_detector():

    #returns a mask-rcnn instance trained on MSCOCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    return model

def get_coco_nouns_from_img(img_paths, device, transform, cat, model):

    for img_path in img_paths:

        img = transform(np.asarray(pil_loader(img_path))).to(device).unsqueeze(0)

    model.eval()
    predictions = model(img)
    obj_indices = list(predictions[0]['labels'].cpu().numpy())

    nouns = []
    for obj_idx in set(obj_indices):
        nouns.append(cat[obj_idx])

    return nouns

def zero_shot_evaluation_nyt(experiment_type=1, template_option=1, name=None, use_encoded_images=False, passed=None):
    model_name = "ViT-B/32"
    log_dir = "www/logs"

    strange_results = []

    if name is None:
        raise Exception("Your experiment must be named")

    log_dir = os.path.join(log_dir, name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    final_stats = open(os.path.join(log_dir, 'stats.txt'), 'w')
    success_at_r1 = open(os.path.join(log_dir, 'success_r1.txt'), 'w')
    success_at_r3 = open(os.path.join(log_dir, 'success_r3.txt'), 'w')
    success_at_r5 = open(os.path.join(log_dir, 'success_r5.txt'), 'w')
    success_at_r10 = open(os.path.join(log_dir, 'success_r10.txt'), 'w')
    failures = open(os.path.join(log_dir, 'failures.txt'), 'w')
    #gtruth = open(os.path.join(log_dir, 'gtruth.txt'), 'w') #records the correct answer to the given index so we can display easily
    #depending on experiment type, from geo/time (given hierarchy) to correct samples

    idx_results = {}
    gtruth_results = {}
    cat_by_class = {}

    if passed['dataset'] is None:

        zero_shot_dataset = GeoTemporalDataset(model_name,
                                     experiment_type,
                                     template_options=(template_option, template_option))

        zero_shot_dataset_info = zero_shot_dataset


    else:

        zero_shot_dataset = passed['dataset']
        zero_shot_dataset_info = zero_shot_dataset.dataset
    #received_keys = sorted(list(zero_shot_dataset.idx_to_url.keys()))

    dataloader = torch.utils.data.DataLoader(dataset=zero_shot_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=custom_collate,
                                             num_workers=10)

    encoded_path = 'encoded-images'
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)


    experiment_path = 'experiments/{}'.format('time' if zero_shot_dataset_info.exp_type == 1 else 'geo')
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)


    current_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if passed['model'] is None:

        model, _ = clip.load(model_name, device=zero_shot_dataset_info.device)

    else:

        model = passed['model']

    total_c1, total_c3, total_c5, total_c10 = 0, 0, 0, 0
    total_all = 0
    with tqdm(dataloader) as t:
        for batch_idx, data in enumerate(t):



            idx = int(data['idx'][0])

            if data['tokenized_correct'].tolist()[0] == -1:

                strange_results.append(data['correct_string'][0])
                continue

            if (not os.path.exists(os.path.join(encoded_path, f'{idx}.pt'))) and use_encoded_images:

                img = data['img'].squeeze(0).to(device=current_device)
                with torch.no_grad():
                    embedding = model.encode_image(img)
                    embedding /= embedding.norm(dim=-1, keepdim=True)
                    torch.save(embedding, os.path.join(encoded_path, f'{idx}.pt'))


                continue


            else:

                if passed['model'] is None and use_encoded_images:
                    embedding = torch.load(os.path.join(encoded_path, f'{idx}.pt'))

                else:
                    img = data['img'].squeeze(0).to(current_device)
                    with torch.no_grad():
                        embedding = model.encode_image(img)
                        embedding /= embedding.norm(dim=-1, keepdim=True)


                tokenized_text = data['tokenized_text'].to(current_device)
                with torch.no_grad():

                    text_embeddings = model.encode_text(tokenized_text.squeeze(0))
                    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

                similarities = (embedding@text_embeddings.T)
                values_top_10, indices_top_10 = similarities[0].topk(10)
                retrieved_values = [data['full_retrieval_set'][idx][0] for idx in indices_top_10]

                sim_values = [float(each) for each in list(values_top_10.cpu().numpy())]
                final_ordering = [(retrieval, similarity) for retrieval, similarity in zip(retrieved_values, sim_values)]
                idx_results[int(idx)] = final_ordering
                gtruth_results[int(idx)] = [data['full_correct_set'][i][0][0] for i in range(len(data['full_correct_set']))]

                top_retrieval = final_ordering[0][0]

                top_obj_info = []
                if experiment_type == 1:

                    dow, day, month, year, decade = recover_time_params(top_retrieval[0], template_option)
                    temp_list = [dow, day, month, year, decade]
                    for time_metric in temp_list:
                        if time_metric is not None:
                            top_obj_info.append(time_metric)



                if experiment_type == 2:

                    if template_option == 1:
                        top_retrieval = top_retrieval[0]

                    if template_option == 2:

                        top_retrieval = re.match("This picture was taken at the location: (.*)", top_retrieval[0])

                        top_retrieval = top_retrieval.group(1)

                    loc = Location(top_retrieval)
                    if loc.country is not None:
                        top_obj_info.append(loc.country)

                    if loc.continent is not None:
                        top_obj_info.append(loc.continent)

                for each in top_obj_info:
                    if each in cat_by_class:

                        cat_by_class[each].append(data['gt_data_url'][0])

                    else:
                        cat_by_class[each] = [data['gt_data_url'][0]]



                indices_top_5 = indices_top_10[:5]
                indices_top_3 = indices_top_10[:3]
                indices_top = indices_top_10[:1]

                '''
                
                '''

                if np.intersect1d(indices_top.cpu(), data['correct_indices']).shape[0] >= 1:
                    success_at_r1.write(f'{idx}\n')
                    total_c1 += 1
                    total_c3 += 1
                    total_c5 += 1
                    total_c10 += 1

                elif np.intersect1d(indices_top_3.cpu(), data['correct_indices']).shape[0] >= 1:

                    success_at_r3.write(f'{idx}\n')
                    total_c3 += 1
                    total_c5 += 1
                    total_c10 += 1

                elif np.intersect1d(indices_top_5.cpu(), data['correct_indices']).shape[0] >= 1:

                    success_at_r5.write(f'{idx}\n')
                    total_c5 += 1
                    total_c10 += 1

                elif np.intersect1d(indices_top_10.cpu(), data['correct_indices']).shape[0] >= 1:

                    success_at_r10.write(f'{idx}\n')
                    total_c10 += 1

                else:
                    failures.write(f'{idx}\n')


                total_all += 1



    final_hit1, final_hit3, final_hit5, final_hit10 =  (total_c1 / total_all) * 100, (total_c3/ total_all) * 100, (total_c5 / total_all) * 100, (total_c10 / total_all) * 100
    final_stats.write(f'Hit@1: {final_hit1}\n Hit@3: {final_hit3}\n Hit@5: {final_hit5}\n Hit@10: {final_hit10}')
    final_stats.close()
    success_at_r1.close()
    success_at_r3.close()
    success_at_r5.close()
    success_at_r10.close()
    failures.close()

    with open(os.path.join(log_dir, 'full_results.json'), 'w') as write_file:
        json.dump(idx_results, write_file)

    with open(os.path.join(log_dir, 'full_gtruth.json'), 'w') as g_write_file:
        json.dump(gtruth_results, g_write_file)

    with open(os.path.join(log_dir, 'sorted_by_class.json'), 'w') as s_write_file:
        json.dump(cat_by_class, s_write_file)

    print(strange_results)

    return final_hit1, final_hit3, final_hit5


def count_labelled_objects():

    model_name = "ViT-B/32"
    zero_shot_dataset = GeoTemporalImage(model_name,
                                     1,
                                     template_options=(1, 1))
    #received_keys = sorted(list(zero_shot_dataset.idx_to_url.keys()))

    dataloader = torch.utils.data.DataLoader(dataset=zero_shot_dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=custom_collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = get_transform()
    cat = get_coco_categories()
    model = get_object_detector().to(device)
    model.eval()

    all_object_counts = {}
    with tqdm(dataloader) as t:
        for batch_idx, data in enumerate(t):

            if batch_idx%10 == 0:
                print(all_object_counts)

            all_imgs = data['img'].to(device=zero_shot_dataset.device).squeeze(1)
            print(all_imgs.shape)

            '''
            all_imgs = []
            for img_path in img_paths:
                img = transform(np.asarray(pil_loader(img_path))).to(device)
                all_imgs.append(img)

            all_imgs = torch.stack(all_imgs)
            '''

            predictions = model(all_imgs)

            for idx in range(len(predictions)):

                obj_indices = list(predictions[idx]['labels'].cpu().numpy())
                nouns = []
                for obj_idx in set(obj_indices):
                    nouns.append(cat[obj_idx])

                for each_object in nouns:

                    if each_object in all_object_counts:
                        all_object_counts[each_object] += 1

                    else:
                        all_object_counts[each_object] = 1

            del all_imgs

    return all_object_counts

def test_location_api():

    strings = [
    "South Africa",
    "Manhattan, New York County, New York, United States",
    "California, United States",
    "Okinawa and Other Ryukyu Islands (Japan)",
    "Nashik",
    ]

    for each in strings:
        loc = Location(each)
        print(loc.country)
        print(loc.continent)
        print(loc.rest)

def quick_clip_call_for_attention(provided_start=0):

    model_name = "ViT-B/32"
    zero_shot_dataset = GeoTemporalImage(model_name,
                                     1,
                                     template_options=(1, 1))
    #received_keys = sorted(list(zero_shot_dataset.idx_to_url.keys()))

    dataloader = torch.utils.data.DataLoader(dataset=zero_shot_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=custom_collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load(model_name, device=zero_shot_dataset.device)
    model = model.to(device)
    model.eval()
    failure_cases = []
    with tqdm(dataloader) as t:
        for batch_idx, data in enumerate(t):

            if batch_idx < provided_start:
                continue

            try:


                img = data['img'].squeeze(0).to(device=device)
                root_path = data['root_img_path']
                save_path = root_path[0].replace('nyt_dataset', 'self_attention').split('.')[0]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                img_cv2 = cv2.resize(cv2.imread(root_path[0]), (img.shape[2], img.shape[3]))


                with torch.no_grad():
                    embedding, attn_weights = model.encode_image(img, get_attn=True)
                    idx = 0
                    for attn in attn_weights:

                        attn /= torch.max(attn)

                        attn = np.uint8(255*attn.cpu().numpy())
                        attn = cv2.resize(attn, (img.shape[2], img.shape[3]))
                        heatmap = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
                        superimposed_img = heatmap * 0.4 + img_cv2

                        cv2.imwrite(os.path.join(save_path, f'sa_{idx}.png'), superimposed_img)

                        idx += 1

                    embedding /= embedding.norm(dim=-1, keepdim=True)

            except:

                failure_cases.append(data['root_img_path'])
                print('model failed here')
                continue




def finetune_full(experiment_type, template_options, jsonl_path=None, epochs=1, mode='train', model_path=None, args=None):


    random.seed(42)
    model_name = "ViT-B/32"

    zero_shot_dataset = GeoTemporalDataset(model_name,
                                     experiment_type,
                                     template_options=(template_options, template_options),
                                     sample=True,
                                     num_negatives=args.num_negatives)


    jsonl_path = None
    if jsonl_path is not None:

        paths = [os.path.join(jsonl_path, each) for each in ['train.jsonl', 'dev.jsonl', 'test.jsonl']]
        train_indices, dev_indices, test_indices, _, _, failures = get_indices_given_split(paths, zero_shot_dataset.url_to_idx)

    else:

        total_indices = len(zero_shot_dataset)
        print(total_indices)

        train_indices = random.sample(range(total_indices), int(0.8 * total_indices))
        remain_indices = set(range(total_indices)).difference(set(train_indices))

        dev_indices = random.sample(remain_indices, int(0.1 * total_indices))
        test_indices = set(remain_indices).difference(set(dev_indices))


    trainset = torch.utils.data.Subset(zero_shot_dataset, train_indices)
    devset = torch.utils.data.Subset(zero_shot_dataset, dev_indices)
    testset = torch.utils.data.Subset(zero_shot_dataset, test_indices)

    if mode=='train':
        trainset.dataset.set_mode('train')
        devset.dataset.set_mode('test')
        finetune_clip_with_dataset(trainset, devset, experiment_type, layers=None, epochs=epochs, args=args)

    else:

        if model_path is None:
            raise Exception("Model path not found")

        else:
            testset.dataset.set_mode('test')
            eval_before_and_after_model(model_path, testset, experiment_type, template_options, args.num_negatives)


def eval_before_and_after_model(model_path, test_dataset, experiment_type, template_options, num_negatives=None):

    if num_negatives is None:
        num_negatives = 19

    model_name = test_dataset.dataset.clip_model_name
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load(model_name, device=device)
    name_of_model = os.path.basename(model_path)

    if not os.path.exists('eval'):
        os.makedirs('eval')
    #final_name_before = f'before_finetune_train_{name_of_model}_test_e{experiment_type}_t{template_options}'
    final_name_after = os.path.join('eval', f'eval_train_{name_of_model}_test_e{experiment_type}_t{template_options}_n{num_negatives}')

    #zero_shot_evaluation_nyt(experiment_type, template_options, passed={'dataset': test_dataset, 'model': model}, name=final_name_before)

    model.load_state_dict(torch.load(model_path))
    zero_shot_evaluation_nyt(experiment_type, template_options,
                             passed={'dataset': test_dataset, 'model': model}, name=final_name_after)



#TODO change naming system inside of model
#TODO change structure inside of dataset class


"""
def untokenize_batch(batch_text, tokenizer):
    
    sentences_in_batch = batch_text.cpu().numpy().tolist()
    decoded_sentences = 0
    for sentence in sentences_in_batch:
        
        current_sentence = ''
        
        for each_word in sentence
        
"""

def finetune_clip_with_dataset(train_dataset, dev_set, experiment_type, layers=None, epochs=1, hparams=None, args=None):


    if hparams is None:
        hparams = {'lr': args.lr, 'batch_size': 32}
    model_name = train_dataset.dataset.clip_model_name
    finetune_dataset = train_dataset
    eval_dataset = dev_set
    batch_size = hparams['batch_size']
    lr = hparams['lr']

    finetune_dataloader = torch.utils.data.DataLoader(
        dataset=finetune_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate,
        num_workers=8,
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load(model_name, device=device, jit=False)
    optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    if layers is None:

        layers = {}
        layers['image'] = []
        layers['text'] = []

        for name, param in model.visual.named_parameters():

            layers['image'].append(name)


        for name, param in model.transformer.named_parameters():

            layers['text'].append(name)


    for name, param in model.visual.named_parameters():

        if name in layers['image']:

            param.requires_grad = True

        else:
            param.requires_grad = False


    for name, param in model.transformer.named_parameters():

        if name in layers['text']:

            param.requires_grad = True

        else:

            param.requires_grad = False



    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    if not os.path.exists(os.path.join('checkpoints_2', args.name)):
        os.makedirs(os.path.join('checkpoints_2', args.name))


    for epoch_no in range(epochs):

        torch.save(model.state_dict(), os.path.join('checkpoints_2', args.name, f'first_finetune_{experiment_type}_{epoch_no}.pth'))

        '''First evaluate and plot'''
        # model.eval()
        dev_set.dataset.set_mode('test')
        # recall_1, recall_3, recall_5, = zero_shot_evaluation_nyt(experiment_type, int(args.to), args.name, use_encoded_images=False, passed={'dataset': eval_dataset, 'model': model})


        if args.train_in_eval:
            model.eval()

        else:
            model.train()

        '''Then train'''
        train_dataset.dataset.set_mode('train')
        avg_loss = 0.0
        with tqdm(finetune_dataloader) as t:

            for batch_idx, data in enumerate(t):


                #print(f'Batch number {batch_idx} is being executed')

                #print(f'The batch contains the following indices:')

                '''
                for each in data['idx'].cpu().numpy().tolist():
                    print(each)
                '''

                optim.zero_grad()

                current_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                img = data['img'].to(current_device).squeeze(1)
                txt = data['correct_string_tokens'].to(current_device).squeeze(1)

                logits_per_image, logits_per_text = model(img, txt)
                gt = torch.arange(img.shape[0], dtype=torch.long, device=current_device)

                total_loss = (loss_img(logits_per_image, gt) + loss_txt(logits_per_text, gt))/2
                total_loss.backward()

                avg_loss += total_loss.item()

                optim.step()


                print(f'Batch number {batch_idx} finished executing')

                if batch_idx == 152:
                    print('Here at stoppage')

            #print('Check profiler outputs at debugging here')
                #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)



        final_avg_loss = avg_loss / len(finetune_dataloader)


    torch.save(model.state_dict(), os.path.join('checkpoints_2', f'first_finetune_{experiment_type}_{epochs}_{args.name}.pth'))
    print('Finished finetuning stage')
