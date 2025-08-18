@echo off

python federatedscope/main.py --cfg federatedscope/glue/yamls/rte/fedex-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/rte/fedsa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/rte/ffa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/rte/lora.yaml

echo. 
pause