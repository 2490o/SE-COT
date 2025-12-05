# Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection

##### Official implementation of ["Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection"], Zihao Zhang, Aming Wu, Yahong Han

![image](pic/f2.jpg)
##  Usage

###  Prepare Data


- **Object Detection**:  
  **Diverse-Weather Dataset**  
  Contains five data/scene conditions:  
  - Daytime-Sunny  
  - Night-Sunny  
  - Dusk-Rainy  
  - Night-Rainy  
  - Daytime-Foggy
 
   **Generalization from Reality to Art**  
  Contains Four conditions:  
  - Real  
  - Clipart  
  - Watercolor  
  - Comic  


### Dependencies

- Python: `3.8.1`  
- PyTorch: `1.10.1`  
- CUDA: `11.8`  
- NumPy: `1.22.4`  
- PIL: `7.2.0`  
- Pillow: `9.5.0`  
- clip: `1.0`  
- detectron2: `0.6`



###  Train and Test

#### Train on Source Domain

CUDA_VISIBLE_DEVICES=0 python train_c.py --config-file configs/diverse_weather_c.yaml

#### Test on Target Domain (e.g., Daytime-Foggy)

python train.py --config-file configs/diverse_weather_foggy_test.yaml --eval-only MODEL.WEIGHTS all_outs/diverse_weather/model_best.pth 
