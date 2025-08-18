@echo off

python federatedscope/main.py --cfg federatedscope/glue/yamls/mmnli/fedex-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/mmnli/fedsa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/mmnli/ffa-lora.yaml
python federatedscope/main.py --cfg federatedscope/glue/yamls/mmnli/lora.yaml

echo.
pause