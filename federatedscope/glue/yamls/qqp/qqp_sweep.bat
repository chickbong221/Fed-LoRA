@echo off

python federatedscope/main.py --cfg federatedscope/glue/yamls/qqp/fedex-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/qqp/fedsa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/qqp/ffa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/qqp/lora.yaml

echo. 
pause