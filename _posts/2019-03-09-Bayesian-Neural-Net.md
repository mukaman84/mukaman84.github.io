---
layout: post
title:  "GPU architecture and CUDA programming"
---
본 페이지에서는 bayesian neural network에 대해서 정리하기위 해사 용되었다.


먼저 GPU의 주된 처리 방식에 대해서 정리해본다.




 
저자의 책  그림 1.13 보면 알 수 있듯이, GPU와 CPU는 일반적으로 DMA(Direct Memory Access)엔진 을 
사용하여  양쪽 메모리 사이의  데이터를 전송하였다. ( 물리적 연결은 어떻게? -- DMA에 대하여 대하여 보다 ㄱ오부할 것)

메모리 공유방법 :
1) DMA
2) PCI Express를 경유 -> 통신 오버헤드가 큼(gpu ->PCI ->보드->CPU 등등)

이 방식들의 문제는 무엇인가?
- GPU와 CPU간에 별도 메모리가 존재하면 깊은 복사로 인해 오버헤드가 상당하다. 따라서 공통의 메모리 공간을 갖추는 것이 중요하나 
다음의 한계로 어렵다.

1) CPU - 대용량 메모리 필요
2) GPU - 고연산 성능을 가능하기 위한 고 대역폭 메모리 필요

-> 즉 gpu 제조사는 이러한 한계속에서 공통 메모리 공간을 다루는 것이 중요한 개발 목표가 됨 -> cpu 제조사도 마찬가지인듯?
예로 CPU 제조사에서 고 대역폭 요구사항을 맞춰 주기 위하여 일반적으로 cpu에서 사용되는 DDR3/4 메모리 경우 대역폭이 부족하여 GDDR DRAM등의 방법으로 메모리 대역폭을 개선중에 있다. 또한 L4 캐시를 장착하여 GPU가 요구하는 높은 메모리 대역을 실현하고 있다.zz








CUDNN은 딥러닝을 위해서 반복적으로 사용되는 컨볼루션, 폴링, signoid, Batchnorm(?)과 같은 기본적인 기능들을 정해진 알고리즘에 따라서 cuda 프로그래밍으로 사전에 구현해 놓은 라이브러리를 가리킨다. 예로 컨볼루션의 경우 gemm, fast gemm, FFT, Winograd 알고리즘이 대표적인 것이다.
아래의 그림은 gemm을 CUDA 프로그래밍으로 구현하는 기본 메카니즘을 보여준다.
![Bayesian_Net](https://cdn-images-1.medium.com/max/1200/1*n6Td0BSmvCGaTYaIJEqF-g.png)
