설명
-
기본적으로 아래의 코드를 따라 작성했습니다.

https://huggingface.co/blog/fine-tune-vit

페이지에 적혀있지 않았지만 재현을 위해 필요한 과정들은 다음과 같습니다.
-

`pip install Pillow`
`pip install torch`
`pip install evaluate`

`pip install sklearn`은 다음과 같은 오류로 인해 scikit-learn을 설치하였음

`pip install scikit-learn`

```
The 'sklearn' PyPI package is deprecated, use 'scikit-learn'
      rather than 'sklearn' for pip commands.
      
      Here is how to fix this error in the main use cases:
      - use 'pip install scikit-learn' rather than 'pip install sklearn'
      - replace 'sklearn' by 'scikit-learn' in your pip requirements files
        (requirements.txt, setup.py, setup.cfg, Pipfile, etc ...)
      - if the 'sklearn' package is used by one of your dependencies,
        it would be great if you take some time to track which package uses
        'sklearn' instead of 'scikit-learn' and report it to their issue tracker
      - as a last resort, set the environment variable
        SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True to avoid this error
      
      More information is available at
      https://github.com/scikit-learn/sklearn-pypi-package
```

`training_args.evaluation_strategy` > `training_args.eval_strategy` // 최신 버전의 변경사항

```
ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`
```
로 인하여 `pip install transformers[torch]`

```
RuntimeError: TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or install tensorboardX.
```
로 인하여 `pip install tensorboardX`


GUI를 위한 과정들
-
`pip install --upgrade gradio`