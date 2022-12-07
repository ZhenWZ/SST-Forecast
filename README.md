# Hierarchical Vector Quantized Variational U-Net for Sea Surface Temperature Forecast

## Model

We proposed a Hierarchical Vector Quantized Variational U-Net model for sea surface temperature prediction. The architecture of our model is shown in the below figure:

![Model](https://user-images.githubusercontent.com/48997918/206292698-3e4900e2-a436-46e9-93ba-847d23ebc832.png)

Our model is based on the V-Unet model implemented in [this repository](https://github.com/iarai/weather4cast#weather4cast-multi-sensor-weather-forecasting-competition--benchmark-dataset). 

## Experiments

The model is fed with 16 frames of historical temperature maps and outputs 4 frames of predicted future temperature maps at once. The dataset is pre-sampled with a sampling interval of 5 to help learn longer time dependency. 

The results are shown as below: 

![Final Presentation](https://user-images.githubusercontent.com/48997918/206293530-09320b21-85b3-4cf6-b916-047648716766.png)

![Results-Hierarchical](https://user-images.githubusercontent.com/48997918/206293546-23694745-928e-4696-ae68-d67e3faa85ec.png)

