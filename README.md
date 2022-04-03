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
We provide the train, dev, and test set in the <i>input</i> folder. In addition, we provide an [html file](https://drive.google.com/file/d/1yVZtFZvtoCc8-3xxpPAsrvvapeaIKR_5/view?usp=sharing) to better demonstrate the dev data.

<details>
  <summary>Show example data in the Development Set.</summary>
  
  ### Heading
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>

## Pretrained and Finetuned CLIP Models ##

## Baseline Models ##

We used the model provided in the original [CLIP repo](https://github.com/openai/CLIP) in our experiments, and used their code to load and predict. Our fine-tuned models can be found [here](https://drive.google.com/drive/folders/1KNcEN3yvhki4XNIfg-t5mXlQZvS1h1XA?usp=sharing).
To use the models, follow the original CLIP repo.

<!-- ## Requirements ## -->

## Citation ##

