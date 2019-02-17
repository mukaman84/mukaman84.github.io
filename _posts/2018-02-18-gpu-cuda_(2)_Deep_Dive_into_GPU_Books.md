---
layout: post
title:  "GPU architecture and CUDA programming"
---
본 페이지에서는 " 머신러닝과 블록체인을 떠받치는 GPU의 모든기술"이라는  도서를 Deep Dive 하여 모두 파헤치기 
위해 작성하였다.


먼저 GPU의 주된 처리 방식에 대해서 정리해본다.
 
저자의 책  그림 1.13 보면 알 수 있듯이, GPU와 CPU는 일반적으로 DMA(Direct Memory Access)엔진 을 
사용하여  양쪽 메모리 사이의  데이터를 전송하였다. ( 물리적 연결은 어떻게?)

PCI Express를 경유하여  






CUDNN은 딥러닝을 위해서 반복적으로 사용되는 컨볼루션, 폴링, signoid, Batchnorm(?)과 같은 기본적인 기능들을 정해진 알고리즘에 따라서 cuda 프로그래밍으로 사전에 구현해 놓은 라이브러리를 가리킨다. 예로 컨볼루션의 경우 gemm, fast gemm, FFT, Winograd 알고리즘이 대표적인 것이다.
아래의 그림은 gemm을 CUDA 프로그래밍으로 구현하는 기본 메카니즘을 보여준다.
![CuDNN](https://www.groundai.com/media/arxiv_projects/11346/genr.svg)
