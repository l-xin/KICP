 # A Knowledge-Injected Curriculum Pretraining Framework for Question Answering
Source code for paper *A Knowledge-Injected Curriculum Pretraining Framework for Question Answering*.

 ## Dependencies
- python >= 3.6

- torch
- transformers

- OpenHowNet

 ## Usage
- Pretrain LM on KG for lesson 1, 2, 3
```bash
python3 src/model/main.py --cuda 0 --dataset fbqa --bert bert-base-uncased --lesson 1 --epoch 3 --batch 32 --lr 5e-4 --fp16 --tag l1
python3 src/model/main.py --cuda 0 --dataset fbqa --bert bert-base-uncased --lesson 2 --epoch 3 --batch 32 --lr 5e-4 --fp16 --tag l2 --saved-mlm l1
python3 src/model/main.py --cuda 0 --dataset fbqa --bert bert-base-uncased --lesson 3 --epoch 3 --batch 32 --lr 5e-4 --fp16 --tag l3 --saved-mlm l2
```
- Finetune and test model on QA dataset
```bash
python3 src/model/main.py --cuda 0 --dataset fbqa --bert bert-base-uncased --qa --epoch 30 --batch 32 --lr 5e-4 --fp16 --tag qa --saved-lm l3
```
- If you hope to enable `torch.nn.DataParallel`(DP), just add more cuda devices
```bash
python3 src/model/main.py --cuda 0,1 ---dataset fbqa
```
- If you hope to enable `torch.nn.parallel.DistributedDataParallel`(DDP), please use the `torchrun` command and add more cuda devices
```bash
torchrun --nnode 1 --nproc_per_node=2 src/model/main.py --cuda 0,1 --dataset fbqa
```

For more running arguments, please refer to [src/model/main.py](src/model/main.py).

## Citation
If you find our work helpful, please consider citing our paper.
```
@inproceedings{lin2024knowledge,
  title={A Knowledge-Injected Curriculum Pretraining Framework for Question Answering},
  author={Lin, Xin and Su, Tianhuang and Huang, Zhenya and Xue, Shangzi and Liu, Haifeng and Chen, Enhong},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={1986--1997},
  year={2024}
}
```
