# Official Implementation of ECCV'24 paper "Bidirectional Uncertainty-Based Active Learning for Open Set Recognition"

by **Chen-Chen Zong, Ye-Wen Wang, Kun-Peng Ning, Hai-Bo Ye, Sheng-Jun Huang**

[[Main paper]](https://arxiv.org/abs/2402.15198) [[Appendix]](https://github.com/chenchenzong/BUAL/blob/main/ECCV2024_BUAL_appendix.pdf) [[Code]](https://github.com/chenchenzong/DPC/blob/main/ECCV2024_BUAL_code/README.md)

## Abstract

Active learning (AL) in open set scenarios presents a novel challenge of identifying the most valuable examples in an unlabeled data pool that comprises data from both known and unknown classes. Traditional methods prioritize selecting informative examples with low confidence, with the risk of mistakenly selecting unknown-class examples with similarly low confidence. Recent methods favor the most probable known-class examples, with the risk of picking simple already mastered examples. In this paper, we attempt to query examples that are both likely from known classes and highly informative, and propose a Bidirectional Uncertainty-based Active Learning (BUAL) framework. Specifically, we achieve this by first pushing the unknown class examples toward regions with high-confidence predictions with our proposed Random Label Negative Learning method. Then, we propose a Bidirectional Uncertainty sampling strategy by jointly estimating uncertainty posed by both positive and negative learning to perform consistent and stable sampling. BUAL successfully extends existing uncertainty-based AL methods to complex open-set scenarios. Extensive experiments on multiple datasets with varying openness demonstrate that BUAL achieves state-of-the-art performance.

## Citation

If you find this repo useful for your research, please consider citing the paper.

```bibtex
@inproceedings{zong2024bidirectional,
  title={Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation},
  author={Zong, Chen-Chen and Wang, Ye-Wen and Ning, Kun-Peng and Ye, Hai-Bo and Huang, Sheng-Jun},
  booktitle={European Conference on Computer Vision},
  pages={127--143},
  year={2024},
  organization={Springer}
}
```
