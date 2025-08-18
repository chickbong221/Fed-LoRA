@echo off

python federatedscope/main.py --cfg federatedscope/glue/yamls/mnli/fedex-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/mnli/fedsa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/mnli/ffa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/mnli/lora.yaml

echo. 
pause