---
layout: post
title:  "얼굴인식과 영상인식 처리"
---
 Face recognition and image processing
---

1강- 얼굴인식, 영상인식 및 처리 산업현황과 국내외 주요 업체 사업 동향
- 라온피플 이석중 대표

(1) rule 기반 비전 검사 및 육안 검사의 한계
 -왜 딥러닝이 적용되었는지 살펴보자

기존 머신비전은 조명등의 환경을 맞추어 제한된 환경에서만 동작하였다.
기존 비전검사 방법은 어려웠다 이유는
    1) 이미지 획득, 전처리
    2) 중간과정에서는 세그멘테이션, 재해석 등 
    3) 해석

각 과정별로 다양한 툴 언어가 존재한다.

비전 검사시 주요 어려운점은 - 각 제품별 불량의 기준이 다를수있다.

다시 말하면, 정상제품하고 모습이 달라도 불량이 아닌 정상일 수 있다.


-> 이러한 어려움은 케이스 바이 케이스로 정의가 어려워, 육안검사 + 머신 비전 검사를 섞어서 사용하였다.

단점
1. 이 경우 작업자의 수준에 따라 다양한 편차가 발생한다.
2. 단순히 육안검사를 위한 인건비 증가 뿐만이 아니라 검사를 위한 인건비 증가등의 비용이 발생한다.
3. 

비전검사 종류
1. 기존 비전검사
2. -> 육안검사
3. -> 딥러닝 비전검사

프루닝 사용이유
- gpu의 비싼 파워 떄문
요새는 자동화지만 fixed-point approximation의 논문을 보면 

hardware-oriented approximatino of convolutional neural networks (2016)
convolution에는 MAC관련 리소스가 엄청 잡아먹는다.
- 최적화하는 즉, 딥러닝 가속기 만드는 사람들은 컨볼루션을 가속화하는 작업을 수행한다.
먼저 fpga로 가속화하는 엔진을 만들고 soc를 양산


자연 영상과 산업 영상의 특성 차이

- 자연: 전 영역에서 색상이 넓게 분포
- 산업: 특정 영역의 색상만이 일정하게 나옴
- 대체적으로 align되어있음
- 이미지 사이즈가 비교적 큼

-> 어떤 방법 사용?


=> 설계 목표 
1) 오픈 소스 대비 파라미터 수 1/100 이하로 감소
2) 대용량 이미지에 대한 실시간 inference가 가능해야한다


여러대의 카메라를 사용했을때 큰 크기의 이미지가 존재
여러각도에서 이미지 촬영을 한 경우 존재
- 여러각도에서 촬영하면서 상태를 classifier
- 멀티 조명을 사용하여 상태를 classifier
   => 이 경우 멀티 채널 이미지로 만들어 네트워크를 돌리면 효율적이었다.

ㄱ기존 네트워크 두고 새로운 네트워크 제안하는 방식으로 신규 불량, 클래스에 대한 결정을 시행


2강- AI 영상 인식 플랫폼 개발과 응용/상용화 동향
- IoT 기반에서 현대인의 생활 환경의 변화

> google glasses 등 다양한 서비스가 존재


1. deep learning with knowledge
=> semantic map -> 아주 중요
비젼 정보와 knowledge를 결합하는 시스템
sensetime의 경우 국가 비전 감시시스템과 유사

2. context-aware service - gps, iot, sensors, automated reasoning and visualization

3. layers of AR Service with IoT

4. AR Techonology Support

5. Technology Maintenance

6. Training with VRA/AR Virtual Experience


3강 - 컴퓨터비전 기반 AI 영상인식 및 처리 방식별 으용 기술개발과 서비스 동향


이력서에 docker, agile process관련 기술 설명할 것
- monocular depth

superdepth
-> super resoultion으로 영상크기 확장하고 이를 이용하여 depth 추정
-> 3D map construction과 mapping까지 수행함

noise2noise
resolution에 대해서 psnr을 추출하고 psnr이 최소화되는 방향으로 네트워크 돌려서 노이즈 최소화

gaugan
2D->3D 변경
- 중요 포인트 : 배치 노말라이제이션
=> data 수집에 한계가 있어 이렇게 만드는 것이 빠를수 있다.

video2video synthesize
=
> frame과 frame사이의 영상을복원


xnor.ai
저가형 하드웨어 업체
- fpga 욜로 알고리즘 동작

binarization으로 convolution 필터를 구현하였고, 이를 저가형 fpga로 컨버팅
차세대 아키텍처 이용


non-local block -> RNN 사용하지도 않고 시퀀스 데이터에서 성능을 향상 시킬 수 있음
(어디서 한겨)

CRF 사용 요령
-> 전처리로 deblur(어떤 deblur 방식이 좋은가?), color normalization
-> 후처리로는 CRF가 너무 거대하기 때문에 부분적으로 사용

4강 - AI 영상인식 얼굴인식 처리 최적 알고리즘
영상인식 동향
- jetson TK1 -> 케플러 시리즈 등등

임베디드로 카페2, tensorRT, tensorflowLITE - android와 iOS전용


모델 압축 예시 4가지
-1. parameter pruning and weight sharing
 2. low-rank factorization
 3. Transferred/compact convolutional filters
 4. Teachers-student model



1) pruning
- uncritical인것들을 제거하는 것
- 처음에 전부다 트레이닝하고. 이후 가지치기를 해서 임계값보다 큰 것들을 살려두는 식으로 반복 수행
- quantization도 이 이안에 포함됨
  -> weight sharing으로 비슷한 애들끼리 클러스터링 후 센트로이드를 구하여, 모든 웨이트들이 이 값만을 사용하는 것
- 최근 방법은 CVPR2018에서 발표된 Pruning-Quantization 기술로 압축하는 것

2) Low-rank factorization
- convolution을 전체를 하는 것이 아닌 depthwise convolution 및 point-wise convolution을 적용하는 것
- 



5강 - 얼굴인식 프레임워크

- 기존 기술은 : 정면 영상을 촬영하는 스태틱 방식이었으면
최근 동향은 비디오에서 얼굴인식을 하는 기술이 연구되고 있다.


- sensetime, megvil;l,


1) 얼굴인식 파이프라인
: 입력 영상이 들어오면 얼굴을 추출
-> nowledge로부터 얼굴 수평방향 조절 등) ->. 특징 추출 -> 매칭 분류

- dataset : FDDB, wide face

얼굴인식 기술
1) Verification,
   gallery based,
   classification


Mega face dataset
FaceNet - 



6강 - continual learning
=> Dynamic하게 계속 하는 방법 -> 일부 뉴론만 학습 또는 네트워크 확장 개념

=> Unsupervised Domain Adaptation
-> 

7강 - 모바일 카메라 기반 기술 설명

=> outfcusing -> 다양한 화질 개선
 -> 거리갑추정에서 먼거리 값을 blur

slam -> 결국 다차원 공간 정보 필요
-> 결국 depth 추출이 중요하다.


필요 코어 기술
1) 이미지 프로세싱 및 전처리
2) 이미지 베이스 퍼셉션
3) 나의 위치 및 포즈 예측


- 주요 설명내용
=> Depth map estimation
   -> 현재 제일 좋은 기술은 stereo Matching Network
     -> Pyramid Stereo Matching Network : feature를 모래시계로 쌓는것


stereo camera는 구현문제가 있어 요새 유행하는 방식은 depth-estimation을 single camera로 하는게 유행

unsupervised monocular depth estimation with left-right coninsistency - cvpr 2017


다초점 카메라 - 카메라 자체로 독특한 multi-array camera를 쓰면 all-focusing 카메라를 얻을 수 있음
-> 거리정보도 다양한 값을 얻을 수 있다.

즉 모바일의 카메라 기종에 따라 다양한 기술 개발 가능

예로 베이어 패턴(RGBG)의 카메라가 아닌 RGB(IR) 카메라를 쓰면 IR부분이 apperture(Exposure)부분이 짧기 때문에 더 많은 영역을 볼수가 있다.

dual pixel phone의 경우 두개의 픽셀 정보를 얻을 수 있기 때문에 하나의 카메라로 거리정보를 얻을 수 있다.
