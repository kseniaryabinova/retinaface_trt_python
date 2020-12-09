# RetinaFace в TensorRT

В этом репозитории мы конвертим PyTorch-модель в ONNX-модель и запускаем TensorRT Engine

# Немного о ONNX

ONNX is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers

# Как запустить конвертацию PyTorch-модели в ONNX-модель

1. Скачать веса нейросети из [репозитория с RetinaFace на PyTorch](https://github.com/biubug6/Pytorch_Retinaface)
2. Конвертировать веса нейросети в ONNX-формат следующим образом:
```bash
# конвертим pytorch-модель в onnx-модель
python3 convert_to_onnx.py

# удаляем ненужные слои, которые сгенерил onnx-конвертер
python3 -m onnxsim retinaface_mobile0.25.onnx retinaface_mobile0.25_smpl.onnx
```   
После конвертации в ONNX переходим в репозиторий с оптимизацией в TRT

# Как запустить TRT-engine из питона

1. Установить TensorRT для питона и для С++. [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
2. Запустить engine-файл следующим образом:
```bash
python3 engine.py
```

На выходе мы получим картинку с найденными лицами

![GitHub Logo](prediction.jpg)

# Ссылки на источники

Веса и модель RetinaFace в PyTorch взяты из этого [репозитория](https://github.com/biubug6/Pytorch_Retinaface) 

1. [ONNX](https://onnx.ai/)
2. [ONNX-simplifier](https://github.com/daquexian/onnx-simplifier)
3. [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

# Дополнительная информация

Время работы сети в PyTorch
```bash
[0] net forward time: 0.2689    whole forward time: 0.3411
[1] net forward time: 0.0069    whole forward time: 0.0619
[2] net forward time: 0.0058    whole forward time: 0.0585
[3] net forward time: 0.0058    whole forward time: 0.0587
[4] net forward time: 0.0058    whole forward time: 0.0584
[5] net forward time: 0.0058    whole forward time: 0.0591
[6] net forward time: 0.0058    whole forward time: 0.0585
[7] net forward time: 0.0058    whole forward time: 0.0585
[8] net forward time: 0.0057    whole forward time: 0.0586
[9] net forward time: 0.0057    whole forward time: 0.0581
```

Время работы сети в TRT из питона
```bash
[0] inference time 0.004984140396118164 sec
[1] inference time 0.004915475845336914 sec
[2] inference time 0.0049092769622802734 sec
[3] inference time 0.004907131195068359 sec
[4] inference time 0.004913806915283203 sec
[5] inference time 0.004911899566650391 sec
[6] inference time 0.0049054622650146484 sec
[7] inference time 0.004909992218017578 sec
[8] inference time 0.00490570068359375 sec
[9] inference time 0.0049016475677490234 sec
```
