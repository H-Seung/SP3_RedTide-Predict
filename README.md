# 적조현상 발생여부 예측 모델
## 파일 설명
- AI_16_한승희_Section3.ipynb : 학습데이터 가공 및 모델 구축
- project3_files/ : db저장 및 웹 구현
- AI_16_한승희_Section3.pdf : 발표자료

## 프로젝트 개요
- 기상청 등 공공데이터와 API를 활용하여 로지스틱 회귀분석과 랜덤포레스트로 적조현상 발생여부 예측
## 프로젝트 배경
- 매년 적조로 인한 수산업 피해가 발생하는 가운데, 공공데이터를 활용하여 적조 발생여부를 예측하는 모델을 마련하고자 함

## 프로젝트 수행과정
![image](https://user-images.githubusercontent.com/114974542/229386749-048e1ac1-458d-4247-b1e8-4c4cf2caef19.png)
![image](https://user-images.githubusercontent.com/114974542/229386861-c350b9c6-9550-423a-886f-da2e4980279d.png)
![image](https://user-images.githubusercontent.com/114974542/229386873-35b19652-2194-4384-8a35-140ac034ae6c.png)
![image](https://user-images.githubusercontent.com/114974542/229386920-1358cbed-f772-456d-827e-ece3f9cc717a.png)

### MySQL 데이터 저장
- 기상청으로부터 3~10일 후의 해상,육상 날씨예보를 API로 가져와 가공후 DB 에 저장   

![image](https://user-images.githubusercontent.com/114974542/229388197-7f97dd91-3853-4765-924c-e1138ccfd9c5.png)
### flask 이용한 웹페이지 구현
- 오늘날짜를 기준으로 3~10 사이의 숫자를 입력하면 해당 일수 뒤의 적조 발생여부를 예측   

![image](https://user-images.githubusercontent.com/114974542/229387655-20ff65b9-9b77-48b1-ba1c-d3a4dbb9fdb7.png)
![image](https://user-images.githubusercontent.com/114974542/229387661-4daaf3c5-18b8-49dc-abf1-62af5cdc01ac.png)
