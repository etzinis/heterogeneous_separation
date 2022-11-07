# Heterogeneous conditional source separation training

Code and data recipes for the paper: 
**Heterogeneous Target Speech Separation** 
Efthymios Tzinis, Gordon Wichern, Aswin Subramanian, Paris Smaragdis, and Jonathan Le Roux

TLDR; The main contribution of this paper is to introduce a novel way of training conditional source separation networks using non-mutually exclusive semantic concepts. We try to make the conditioned models mimic humans' flexibility when selecting which source to attend to, by focusing on extracting sounds based on semantic concepts and criteria of different nature, i.e., heterogeneous, such as whether a speaker is near or far from the microphone, being soft or loud, or speaks in a certain language

[![YouTube HCT presentation](https://www.youtube.com/watch?v=tPjGSuBcGA4/0.jpg)](https://www.youtube.com/watch?v=tPjGSuBcGA4 "Virtual Interspeech paper's presentation")

arXiv: https://arxiv.org/abs/2204.03594
pdf: https://arxiv.org/pdf/2204.03594.pdf
slides: https://docs.google.com/presentation/d/15dUDG-qiX0QABBeGZ1f7pmZwtGshSp6w/edit?usp=sharing&ouid=106534973647151270598&rtpof=true&sd=true

Please cite as:
```BibTex
@inproceedings{tzinis22_interspeech,
  author={Efthymios Tzinis and Gordon Wichern and Aswin Shanmugam Subramanian and Paris Smaragdis and Jonathan {Le Roux}},
  title={{Heterogeneous Target Speech Separation}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={1796--1800},
  doi={10.21437/Interspeech.2022-10717}
}
```

## Table of contents

- [Datasets Preparation](#dataset-generation)
- [How to use the dataset loaders](#how-to-run)
- [References](#references)
- [Copyright and license](#copyright-and-license)

We will soon publish the data recipes for: 

- WSJ0-2mix (WSJ0): gender and energy (louder/softer) conditions.
- Spatial LibriSpeech (SLIB): spatial location of the speakers (near/far field), gender, and energy conditions.
- Spatial VoxForge (SVOX): language of the speaker (German/English/French/Spanish), spatial location of the speakers, energy conditions.   
