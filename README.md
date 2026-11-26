# ReworkVLM

**What Should Be Fixed? Rule-Grounded Corrective Actions for Reworkable Logical Anomalies in Industrial Inspection**

Haitao Zhang, Bosheng Chen, Haote Xu, Xiaotong Tu, Yue Huang, and Xinghao Ding

ReworkVLM diagnoses logical anomalies and suggests a minimal corrective action grounded in the normal rule. LOCO-Correct and VID-AD-Correct provide per-image annotations and tasks for anomaly diagnosis, structured action prediction, and open-ended correction. ReworkVLM uses supervised fine-tuning followed by RACO-GRPO on Qwen3-VL-4B-Instruct.

## Release links

| Resource | Hugging Face | ModelScope |
|---|---|---|
| ReworkVLM annotations and splits | [Dataset](https://huggingface.co/datasets/zhtstar/ReworkVLM-Correct) | [Dataset](https://modelscope.cn/datasets/zhtstar/ReworkVLM-Correct) |
| ReworkVLM-4B-LOCO | [Model](https://huggingface.co/zhtstar/ReworkVLM-4B-LOCO) | [Model](https://modelscope.cn/models/zhtstar/ReworkVLM-4B-LOCO) |
| ReworkVLM-4B-VID-AD | [Model](https://huggingface.co/zhtstar/ReworkVLM-4B-VID-AD) | [Model](https://modelscope.cn/models/zhtstar/ReworkVLM-4B-VID-AD) |
| ReworkVLM-4B-Joint | [Model](https://huggingface.co/zhtstar/ReworkVLM-4B-Joint) | [Model](https://modelscope.cn/models/zhtstar/ReworkVLM-4B-Joint) |

The LOCO and VID-AD checkpoints correspond to the dataset-specific routes. Joint training uses the training data from both benchmarks. Each checkpoint already contains the merged SFT and RL update and loads as a standalone model.

## Pair the annotations with source images

1. Download and extract [MVTec LOCO AD from MVTec](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) and [VID-AD from its authors](https://github.com/nkthiroto/VID-AD#download). The LOCO root directly contains `breakfast_box/`; the VID-AD root directly contains `Ropes/` and other category folders. Keep the original paths and capitalization.
2. Download the ReworkVLM-Correct dataset snapshot. It contains JSONL annotations, VQA records, and the frozen SFT, RL and evaluation membership. Source image files come from their respective providers.
3. Run the helper with Python 3.10 or newer. Omit `--output` to check the source-image index only. Use `--file` with `--output` to materialize one JSONL.

```bash
python scripts/pair_images.py --bundle ./ReworkVLM-Correct \
  --loco-root ./images/mvtec_loco_anomaly_detection \
  --vid-root ./images/VID-AD --output ./paired
```

Use `training_inputs.json` from the dataset snapshot to select the correct files in `paired/`. In the image index, `release_split` identifies the benchmark splits. For VID-AD-only, `vid_ad_only_split` also records the internal training and validation assignment within the 8,300-image training pool. The dataset card lists the expected counts and upstream terms.

## Inference and inspection

```bash
pip install torch torchvision 'transformers==4.57.6' accelerate pillow
python scripts/infer.py --model zhtstar/ReworkVLM-4B-LOCO \
  --image ./image.png --prompt "Identify the violated rule and suggest the smallest repair."
```

Select a checkpoint matching your benchmark or use the Joint checkpoint for the combined training route. For custom review work, see the [Flask annotation quality review tool](quality_review/README.md). It loads local images and model drafts, records individual reviews and adjudications, and exports a JSONL decision history.

## Citation

```bibtex
@inproceedings{zhang2026should,
  title = {What Should Be Fixed? Rule-Grounded Corrective Actions for Reworkable Logical Anomalies in Industrial Inspection},
  author = {Zhang, Haitao and Chen, Bosheng and Xu, Haote and Tu, Xiaotong and Huang, Yue and Ding, Xinghao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  pages = {4349--4356},
  year = {2026},
  doi = {10.18653/2026.emnlp-industry.258},
  url = {https://aclanthology.org/2026.emnlp-industry.258/}
}
```
