# Pycon Tutorial : 음악과 딥러닝의 만남
딥러닝과 음악이 만난다면 어떤 Task를 처리할 수 있을까요? Speech 데이터와는 또 다른 매력을 가지고 있는 음악!

과연 음악데이터는 어떤 특징을 가지고 있고 이를 이용해서 우리는 또 어떤 멋진 문제를 해결할 수 있을까요?? 
먼저 우리가 Music Data로 부터 어떤 정보를 얻을 수 있는지 알아보고 과연 학계에서는 어떤 Task를 풀고 있는지를 확인해보려고합니다. 
마지막으로는 Pytorch를 이용하여 아주 간단한 아키텍쳐를 구현해 보고자합니다! 
Python, Pytorch, 그리고 Librosa 함께 우리 음악 데이터를 분석해 봅시다!

### Contents
- (2:00~3:00) Sound Data의 이해와 전처리와 시각화
- (3:00~3:30) Music-Deep Learing 의 주요 Task Review
- (3:30~3:40) Break
- (3:40~5:00) Music Gerne Classification, Music Generation (Optional)

### Requirement
- Python 3.7 (recommended), 3.6 (Good)
- Numpy
- Librosa
- PyTorch 1.0
- 데이터셋을 받을 1GB 이상의 여유공간이 있는 노트북

### 관련 자료
- https://github.com/Dohppak/Music_Genre_Classification_Pytorch
- https://github.com/Dohppak/Music_Generation_Pytorch (Optional)
### Dataset
[GTZAN Subset](https://drive.google.com/file/d/1rHw-1NR_Taoz6kTfJ4MPR5YTxyoCed1W/view)
```
$ cd gtzan
$ ls 
blues disco metal ... rock train_filtered.txt valid_filtered.txt test_filtered.txt
$ cd ..      # go back to your home folder for next steps
```

### Reference

이 자료는 KAIST 남주한 교수님의 강의 GCT634 (음악의 머신러닝적 활용) HW2의 코드를 참고했음을 밝힙니다.
- https://github.com/juhannam/gct634-2019/tree/master/hw2