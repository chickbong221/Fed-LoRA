@echo off

python federatedscope/main.py --cfg federatedscope/glue/yamls/sst2/fedex-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/sst2/fedsa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/sst2/ffa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/sst2/lora.yaml

echo.
pause