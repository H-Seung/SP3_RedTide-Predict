# Section3-Project
> Section3-Project : 적조현상 발생여부 예측 모델
- AI_16_한승희_Section3.ipynb : 학습데이터 가공 및 모델 구축
- project3_files/ : db저장 및 웹 구현
## 예측모델 구축
- 기상청 등 공공데이터를 활용하여 로지스틱 회귀분석과 랜덤포레스트로 적조현상 발생여부 모델을 구축    

![image](https://user-images.githubusercontent.com/114974542/229386698-529ae633-d3a7-4013-9a56-5167af5e985a.png)
![image](https://user-images.githubusercontent.com/114974542/229386749-048e1ac1-458d-4247-b1e8-4c4cf2caef19.png)
![image](https://user-images.githubusercontent.com/114974542/229386861-c350b9c6-9550-423a-886f-da2e4980279d.png)
![image](https://user-images.githubusercontent.com/114974542/229386873-35b19652-2194-4384-8a35-140ac034ae6c.png)
![image](https://user-images.githubusercontent.com/114974542/229386920-1358cbed-f772-456d-827e-ece3f9cc717a.png)
## MySQL 데이터 저장
- 기상청으로부터 3~10일 후의 해상,육상 날씨예보를 API로 가져와 가공후 DB 에 저장   

![image](https://user-images.githubusercontent.com/114974542/229388197-7f97dd91-3853-4765-924c-e1138ccfd9c5.png)
## flask 이용한 웹페이지 구현
- 오늘날짜를 기준으로 3~10 사이의 숫자를 입력하면 해당 일수 뒤의 적조 발생여부를 예측   

![image](https://user-images.githubusercontent.com/114974542/229387655-20ff65b9-9b77-48b1-ba1c-d3a4dbb9fdb7.png)
![image](https://user-images.githubusercontent.com/114974542/229387661-4daaf3c5-18b8-49dc-abf1-62af5cdc01ac.png)
