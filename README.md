
# SAC-LSTM

This is a Pytorch implementation of SAC-LSTM, a recurrent model for radar echo extrapolation (precipitation nowcasting)



# Setup

Required python libraries: torch (>=1.3.0) + opencv + numpy + scipy (== 1.0.0) + jpype1.
Tested in ubuntu + nvidia Titan with cuda (>=10.0).

# Datasets

We conduct experiments on CIKM AnalytiCup 2017 datasets: [CIKM_AnalytiCup_Address](https://tianchi.aliyun.com/competition/entrance/231596/information) or [CIKM_Rardar](https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y?usp=sharing) 

# Training

Use 'train.py' script to train these models. 

You might want to change the parameter and setting, you can change the details of variable ‘args’ 


# Test
Use 'test.py' script to test these models. 

# Prediction samples
5 frames are predicted given the last 10 frames.




