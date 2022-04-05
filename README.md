# TARA: There’s a Time and Place for Reasoning Beyond the Image
<p float="left">
<img src="https://user-images.githubusercontent.com/31453585/153289234-12495801-6dbc-4660-b513-f18b1a16dc42.jpg" alt="new-york-protest-floyd" width="300">
<img src="https://user-images.githubusercontent.com/31453585/153289545-0df21366-644b-488d-8835-1c69e8a5a0d4.jpg" alt="modi-board" width="300"/>
 Can you tell the time and location when the images were taken?
</p>
 

In this work, we identify and formulate this problem, spatio-temporal grounding of images, a task aiming at identifying the time and location the given image was taken. Specifically, we develop a novel dataset TARA, (Time and plAce for Reasoning beyond the imAge), a challenging and important dataset with 16k images with their associated news, time and location automatically extracted from New York Times (NYT), and an additional 61k examples as distant supervision from WIT. On top of the extractions, we present a crowdsourced subset in which images are believed to be feasible to find their spatio-temporal information for evaluation purpose. We show that there exists a  gap between a state-of-the-art joint model and human performance, which is slightly filled by our proposed model that uses segment-wise reasoning, motivating higher-level vision-language joint models that can conduct open-ended reasoning with world knowledge.


In this repository, we provide the dateset for TARA, along with the pytorch implementation of the baseline variants models.


## Datasets ##
Download [here](https://drive.google.com/drive/folders/1KNcEN3yvhki4XNIfg-t5mXlQZvS1h1XA?usp=sharing).
We provide the train, dev, and test set in the <i>input</i> folder. In addition, we provide an [html file](https://drive.google.com/file/d/1yVZtFZvtoCc8-3xxpPAsrvvapeaIKR_5/view?usp=sharing) to better demonstrate the data.
<!-- 
<details>
  <summary>Show example data in the Development Set.</summary>
  
  ### Heading
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details> -->

## Finetuned CLIP Models ##

We used the model 'ViT-B/32' provided in the original [CLIP repo](https://github.com/openai/CLIP) in our experiments, and used their code to load and finetune. 
Please make sure you satisfy all the requirements following the original [CLIP repo](https://github.com/openai/CLIP).

Our fine-tuned models can be found [here](https://drive.google.com/drive/folders/1KNcEN3yvhki4XNIfg-t5mXlQZvS1h1XA?usp=sharing). The fine-tuning code are in ```time-reasoning```.

To evaluate any CLIP-based model on our dataset, you can use the command
```
python eval.py --clip_model_name /YOUR_PATH_TO_SAVE_FINETUNED_CLIP_MODELS/finetune_segment_joint.pth --label_name gold_time_suggest
```


Please note that there are four kinds of labels: [gold_location,	gold_location_suggest; gold_time,	gold_time_suggest] in our dataset. In all our experiments, <b> we use gold_location_suggest and gold_time_suggest only</b>. 

The only difference between gold_LABEL and gold_LABEL_suggest is the granularity. gold_LABEL_suggest is the adjusted gold_LABEL after MTurk annotation. For example, the gold_LABEL is the most precise label we got, e.g. 2017-5-23. Then, during our MTurk annotation process, our human annotators may find that they can only reason the time to a year but not a specific date, so in this case, the gold_LABEL_suggest will become 2017. 

<!-- ## Requirements ## -->

## Citation ##

