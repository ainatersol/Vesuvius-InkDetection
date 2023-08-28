One discrepancy between the kaggle competition data and the additional scroll data is the resolution of the scans. it is an open question whether or not our models will transfer well to this new data. Initial attempts to apply our winning solution to the monster fragment did not look particularly promising. 
![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/Screen_Shot_2023-06-20_at_12.09.32_AM.png?raw=true)I tried various different forms of this. A model trained against 4 micrometer kaggle data and applied to the 8 micrometer scroll data, then retried on the 8 micrometer data upscaled to 4 micrometer on the x,y axis and then again trying to upscale all axis to 4 micrometer. All of them seemed to return something that looked mostly like noise. Our final solution included a handful of models as well as test-time augmentation(doing inference against the same data rotated and flipped multiple times and then averaging the results). I attempted with a range of different models as well as our usual ensemble. Against the monster fragment even with a triple gpu system this took a couple hours. I also tried inference with additional depth layers. Our competition solution only included the middle 16 layers of data, but some of our models were actually easily capable of having a variable number of input layers that were later pooled away so I tried with 32 input layers as well thinking that maybe the less precise segmentations had made them shift out of range but this did not yield anything more than noise as well.

Viewing the distribution of the output of the models I found that they looked significantly differently than our validation predictions looked like. On the new scroll it seemed like there was hardly any signal passing over into the positives. The model did not feel like it was seeing any ink. 

Distribution of raw predictions out of our model. A decent mass of pixels crossing the threshold we set at 0. Anything above 0 is likely ink

![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/validation_distribution.png)


test(monster scroll) distribution. We can see that hardly any cross into the positives

![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/test_distribution.png)

Not finding any success with any of these combinations I went back to the drawing board just to understand exactly what impact the resolution shift had. I took our models that were trained on 4 micrometer and then I did inference on the fragment 1 image that had been held out during our training and compared the output when I gave the model 8 micrometer and 4 micrometer inputs. The output was clearly worse and significantly coarser, but it didnt look completely decayed. It wasn't like the model was outputting noise like in the monster scroll output I was playing with previously.

Original validation predictions for fragment 1 at 4 micrometer input resolution. Raw and thresholded predictions

![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/raw_full_res.png?raw=true)

![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/thresholded_full_res.png?raw=true)

validation predictions for fragment 1 at 8 micrometer input resolution. Raw and thresholded predictions

![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/raw_half_res.png?raw=true)

![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/thresholded_half_res.png?raw=true)


After seeing that the output was decayed but not ruined I tried applying additional models to the monster scroll, both recto and verso sides. Eventually, I was able to get what seemed like a couple letter correctly lined up, but I am still not completely convinced these are true findings or just coincidental hallucinations because other models that were more accurate during the competition did not yield a similar pattern. One point to it's credit though, the model is given tiles of input and this is showing consistency across these tiled boundaries so maybe there is actually something there. 

Prediction against the entire monster fragment
![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/monster_prediction.png)

Small region that looks potentially like a couple letters
![image](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/possible_letters.png)


To check if I could visually see anything in the letters or even if I could find some discrepancies in intensity or some other anomaly between the two images I visualized the middle 16 layers of each piece in these videos

Fragment 1's 16 layer video

https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/frag_1_16_layers.mp4

Monster fragment 16 layer video

https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/monster_frag_16_layers.mp4

16 layer close up of possible letters

https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/close_up_possible_letters.mov

