설명
-
기본적으로 아래의 코드를 따라 작성했습니다.

https://huggingface.co/blog/fine-tune-vit

테스트 환경
-
ubuntu 2204\
python 3.13.2

`pip install datasets Pillow torch evaluate scikit-learn transformers[torch] tensorboardX gradio`

```
Package                  Version
------------------------ -----------
accelerate               1.8.1
aiofiles                 24.1.0
aiohappyeyeballs         2.6.1
aiohttp                  3.12.13
aiosignal                1.3.2
annotated-types          0.7.0
anyio                    4.9.0
attrs                    25.3.0
audioop-lts              0.2.1
certifi                  2025.6.15
charset-normalizer       3.4.2
click                    8.2.1
datasets                 3.6.0
dill                     0.3.8
evaluate                 0.4.4
fastapi                  0.115.13
ffmpy                    0.6.0
filelock                 3.18.0
frozenlist               1.7.0
fsspec                   2025.3.0
gradio                   5.34.2
gradio_client            1.10.3
groovy                   0.1.2
h11                      0.16.0
hf-xet                   1.1.5
httpcore                 1.0.9
httpx                    0.28.1
huggingface-hub          0.33.1
idna                     3.10
Jinja2                   3.1.6
joblib                   1.5.1
markdown-it-py           3.0.0
MarkupSafe               3.0.2
mdurl                    0.1.2
mpmath                   1.3.0
multidict                6.5.1
multiprocess             0.70.16
networkx                 3.5
numpy                    2.3.1
nvidia-cublas-cu12       12.4.5.8
nvidia-cuda-cupti-cu12   12.4.127
nvidia-cuda-nvrtc-cu12   12.4.127
nvidia-cuda-runtime-cu12 12.4.127
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.2.1.3
nvidia-curand-cu12       10.3.5.147
nvidia-cusolver-cu12     11.6.1.9
nvidia-cusparse-cu12     12.3.1.170
nvidia-cusparselt-cu12   0.6.2
nvidia-nccl-cu12         2.21.5
nvidia-nvjitlink-cu12    12.4.127
nvidia-nvtx-cu12         12.4.127
orjson                   3.10.18
packaging                25.0
pandas                   2.3.0
pillow                   11.2.1
pip                      25.1
propcache                0.3.2
protobuf                 6.31.1
psutil                   7.0.0
pyarrow                  20.0.0
pydantic                 2.11.7
pydantic_core            2.33.2
pydub                    0.25.1
Pygments                 2.19.2
python-dateutil          2.9.0.post0
python-multipart         0.0.20
pytz                     2025.2
PyYAML                   6.0.2
regex                    2024.11.6
requests                 2.32.4
rich                     14.0.0
ruff                     0.12.0
safehttpx                0.1.6
safetensors              0.5.3
scikit-learn             1.7.0
scipy                    1.16.0
semantic-version         2.10.0
setuptools               78.1.1
shellingham              1.5.4
six                      1.17.0
sniffio                  1.3.1
starlette                0.46.2
sympy                    1.13.1
tensorboardX             2.6.4
threadpoolctl            3.6.0
tokenizers               0.21.2
tomlkit                  0.13.3
torch                    2.6.0
tqdm                     4.67.1
transformers             4.52.4
triton                   3.2.0
typer                    0.16.0
typing_extensions        4.14.0
typing-inspection        0.4.1
tzdata                   2025.2
urllib3                  2.5.0
uvicorn                  0.34.3
websockets               15.0.1
wheel                    0.45.1
xxhash                   3.5.0
yarl                     1.20.1
```

출력
-
`main.py`
```
Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.
bias', 'classifier.weight']                                                                                                                                           You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/home/<user>/fine-tune-vit/train.py:8: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` ins
tead.                                                                                                                                                                   return Trainer(
{'loss': 0.8089, 'grad_norm': 1.6200957298278809, 'learning_rate': 0.0001930769230769231, 'epoch': 0.15}                                                              
{'loss': 0.3478, 'grad_norm': 2.1580088138580322, 'learning_rate': 0.0001853846153846154, 'epoch': 0.31}                                                              
{'loss': 0.2252, 'grad_norm': 1.9345530271530151, 'learning_rate': 0.0001776923076923077, 'epoch': 0.46}                                                              
{'loss': 0.1988, 'grad_norm': 1.5985360145568848, 'learning_rate': 0.00017, 'epoch': 0.62}                                                                            
{'loss': 0.2318, 'grad_norm': 1.212094783782959, 'learning_rate': 0.0001623076923076923, 'epoch': 0.77}                                                               
{'loss': 0.2389, 'grad_norm': 1.2101879119873047, 'learning_rate': 0.00015461538461538464, 'epoch': 0.92}                                                             
{'loss': 0.1153, 'grad_norm': 0.3894956707954407, 'learning_rate': 0.00014692307692307693, 'epoch': 1.08}                                                             
{'loss': 0.0996, 'grad_norm': 0.15946601331233978, 'learning_rate': 0.00013923076923076923, 'epoch': 1.23}                                                            
{'loss': 0.0668, 'grad_norm': 3.166346549987793, 'learning_rate': 0.00013153846153846156, 'epoch': 1.38}                                                              
{'loss': 0.0732, 'grad_norm': 1.0970507860183716, 'learning_rate': 0.00012384615384615385, 'epoch': 1.54}                                                             
{'eval_loss': 0.07936842739582062, 'eval_accuracy': 0.9774436090225563, 'eval_runtime': 1.1153, 'eval_samples_per_second': 119.255, 'eval_steps_per_second': 15.243, '
epoch': 1.54}                                                                                                                                                         {'loss': 0.0731, 'grad_norm': 8.524600982666016, 'learning_rate': 0.00011615384615384617, 'epoch': 1.69}                                                              
{'loss': 0.1064, 'grad_norm': 0.163397878408432, 'learning_rate': 0.00010846153846153846, 'epoch': 1.85}                                                              
{'loss': 0.0433, 'grad_norm': 0.1583067923784256, 'learning_rate': 0.00010076923076923077, 'epoch': 2.0}                                                              
{'loss': 0.1359, 'grad_norm': 2.322096347808838, 'learning_rate': 9.307692307692309e-05, 'epoch': 2.15}                                                               
{'loss': 0.0717, 'grad_norm': 0.2557260990142822, 'learning_rate': 8.538461538461538e-05, 'epoch': 2.31}                                                              
{'loss': 0.0621, 'grad_norm': 4.89802885055542, 'learning_rate': 7.76923076923077e-05, 'epoch': 2.46}                                                                 
{'loss': 0.0395, 'grad_norm': 0.08469036221504211, 'learning_rate': 7e-05, 'epoch': 2.62}                                                                             
{'loss': 0.0405, 'grad_norm': 7.634711265563965, 'learning_rate': 6.23076923076923e-05, 'epoch': 2.77}                                                                
{'loss': 0.033, 'grad_norm': 0.06909171491861343, 'learning_rate': 5.461538461538461e-05, 'epoch': 2.92}                                                              
{'loss': 0.0474, 'grad_norm': 0.06666595488786697, 'learning_rate': 4.692307692307693e-05, 'epoch': 3.08}                                                             
{'eval_loss': 0.050346605479717255, 'eval_accuracy': 0.9924812030075187, 'eval_runtime': 0.5881, 'eval_samples_per_second': 226.134, 'eval_steps_per_second': 28.904, 
'epoch': 3.08}                                                                                                                                                        {'loss': 0.0187, 'grad_norm': 8.051018714904785, 'learning_rate': 3.923076923076923e-05, 'epoch': 3.23}                                                               
{'loss': 0.0143, 'grad_norm': 0.13006502389907837, 'learning_rate': 3.153846153846154e-05, 'epoch': 3.38}                                                             
{'loss': 0.0117, 'grad_norm': 0.07026470452547073, 'learning_rate': 2.384615384615385e-05, 'epoch': 3.54}                                                             
{'loss': 0.0129, 'grad_norm': 0.06242894008755684, 'learning_rate': 1.6153846153846154e-05, 'epoch': 3.69}                                                            
{'loss': 0.0116, 'grad_norm': 0.06449420750141144, 'learning_rate': 8.461538461538462e-06, 'epoch': 3.85}                                                             
{'loss': 0.013, 'grad_norm': 1.1909167766571045, 'learning_rate': 7.692307692307694e-07, 'epoch': 4.0}                                                                
{'train_runtime': 32.2448, 'train_samples_per_second': 128.269, 'train_steps_per_second': 8.063, 'train_loss': 0.12081946684763982, 'epoch': 4.0}                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 260/260 [00:32<00:00,  8.06it/s]
***** train metrics *****
  epoch                    =         4.0
  total_flos               = 298497957GF
  train_loss               =      0.1208
  train_runtime            =  0:00:32.24
  train_samples_per_second =     128.269
  train_steps_per_second   =       8.063
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17/17 [00:00<00:00, 35.67it/s]
***** eval metrics *****
  epoch                   =        4.0
  eval_accuracy           =     0.9925
  eval_loss               =     0.0503
  eval_runtime            = 0:00:00.53
  eval_samples_per_second =    246.339
  eval_steps_per_second   =     31.487
```

`test.py`
```
./dataset/test.png infered as bean_rust
./dataset/test2.jpg infered as bean_rust
========== validation set (133개) ==========
[validation] misclassificated angular_leaf_spot as bean_rust
accuracy of validation set: 99.24812030075188%
132 out of 133
========== test set (128개) ==========
[test] misclassificated angular_leaf_spot as bean_rust
[test] misclassificated bean_rust as angular_leaf_spot
[test] misclassificated bean_rust as angular_leaf_spot
[test] misclassificated bean_rust as angular_leaf_spot
accuracy of test set: 96.875%
124 out of 128
========== train set (1034개) ==========
accuracy of train set: 100.0%
1034 out of 1034
```