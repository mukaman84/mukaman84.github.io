---
layout: post
title:  "Bayesian Neural Net Introduction"
---
 Baysian Nerural Networks
---

본 포스트에서는 Baysian Nerural Networks에 대해서 기초부터 최신 논문 동향까지를 기술한다.

먼저 기본적인 설명은 [Bayesian Neural Network][1] post를 참고하여 재구성하였다.

[Bayesian Neural Network][1] postsms 8개의 post를 통해서 설명하고 있다. 이 중 본 post는 첫 번째인 Need for Bayesian Neural Networks [Bayesian Neural Network][1]를 주로 참조하였다.

Baysian Nerural Networks를 이해하기 위해 먼저 point-estimate에 대해서 이해할 필요가 있다.
Point-estimate : 아래 그림의 왼쪽 네트워크 예시와 같이 weight(filter의 각 element)의 가 sinle point로 표현되는 것을 의미한다.
Baysian Nerural Network: Point-estimate와는 다르게 아래의 그림에서 오른쪽 네트워크와 같이 weight가 확률의 형태로 표기됨을 의미한다.

그럼 Point-estimate는 왜 문제인가? Over-fitting 때문
- 결국 최종단에서 softmax가 각 pixel 또는, 이미지들의 class를 결정하는데 있어, 하나의 class외에 나머지를 squish하여 zero 근처로 보내버리고 오직 하나만을 maximize하는 방향으로 가버리는데 이는 one class에 대한 overconfident 결정일 수 있다. 특히 imbalanced dataset에서 이러한 overconfident 결정은 더 두드러지게 나타난다.
 즉, 동물을 classfication하는 문제라고하면 개일 확률 0.9로 만들고 나머지는 0.1 이하로 만드는 결정을 할 수 있다.
 때로는 강아지 0.4 늑대 0.3 고양이 0.2로 inference하는 방식이 네트워크의 overfit를 방지할 수 있다.
 -> 그럼 뒤에 결정은 어떻게 하는가? -> 좀 더 고민해 보자
  
![Bayesian_Net](https://cdn-images-1.medium.com/max/1200/1*n6Td0BSmvCGaTYaIJEqF-g.png)

그런데 Overfitting or making overconfident decisions문제를 풀기위해 많은 regularization techniques들이 존재한다.
이들과 Baysian Nerural Network의 차이는 무엇인가? -> 기존 regularization techniques은 결정된 정보의 불확실함을 표현하지 못한다.
그러나 Baysian Nerural Network는 결정된 부분에서 불확실함이 네트워크에 표현이 가능하다.


## The practicality of Bayesian neural networks
이 파트는 Bayesian neural networks를 어떻게 구현할 것인가에 대한 얘기로, 네트워크의 여러 파라미터 (weights, activation results등) 중에서 무엇을 확률 모델로 바꿀것 인가를 다루는 문제이다. 가장 쉽게 생각할 수 있는건 weights를 확률 모델로 가져가는 것이다. 그러나 여태까지 그 누구도 weights를 distribution 모델로 가져가서 성공한 사례는 없다. 그 이유는 너무나 많은 weight 수와 모델의 크기 때문이 아닐까라고 예측되고있다.[1]

그러한 이유로 Bayesian neural networks 연구를 시간의 순서대로 살펴볼 필요가 있다. 초기 연구는 가능성을 보기 위해 FC 구조의 neural network에서 model posterior에 대해 근사화하는 것을 시작으로 한다. 초기에는 posterior 확률로 가우시안 분포와 같은 simple variational distribution을 사용하였고, 네트워크가 학습을 통해서 이러한 true posterior probability에 가까운 분포를 만들어 내도록 학습되었다. 이를 위해 확률 분포간의 가격을 최소화하는 KL-divergence가 사용되었다. 

그러나 이 연구의 문제는 gaussian 근사화로 인해 model parameter들의 수 증가가 너무나 커져서 계산적으로 비싸다는 것이다. 예로 가우시안 근사화에 사용된 파라미터만으로 모델 파라미터들의 두배가 된다는 것이다.

->> 검토 필요 : 그런데 dropout을 사용한 전통적인 방식이 동일한 predictive performance를 보여준다는 결과가 나왔다???

이는 모델의 모든 파라미터들은 변수가 될 수 있고, 이 각각의 변수들을 bayesian 확률 모델로 정의하는 것은 엄청난 계산 복잡도를 가져온다는 것이다. 그래서 Bayesian neural networks 모델을 만드는 수 많은 방법들이 존재하는데 이 포스트에서는 backpropagation에 bayes 이론을 적용하는 것을 집중적으로 살펴본다.

## Bayes by Backprop

Bayes by backprop는 [Blundell, et al.][3]에 의해서 처음 도입되었다. 이 논문에서는 gradient의 unbiased Monte Carlo estimates를 사용한 앙상블들로 네트워크의 파라미터들을 학습시켰다.

그런데 네트워크 파라미터가 점점 증가함에 따라 정확하게 학습시키는 것은 불가능하였다. 
이에 대한 해결책으로 제시된 방법은 최근에 사용되는 variational approximation 기술이다.
기존 Monte Carlo 기술은 너무나 많은 앙상블이 요구되기에 Bayesian posterior distribution을 approximate시키는 것은 정말 어렵기 때문이다.

왜 학습이 어려운지는 [Bayesian neural networks][2] 2번 째 blog post에 다음과 같이 잘 설명되어있다.

- Bayes theorem은 다음과 같다.
![Bayes theorem](https://cdn-images-1.medium.com/max/800/1*7iOrI5jb6Dae630hCYENjA.png)
일반적으로 supervised NN은 data x가 주어질 경우 model parameters θ를 찾는 문제로, 이로부터 posterior probability P(θ|x)가 정의된다.
여기서 P(θ) 는 prior이며, P(x|θ) which is the likelihood로 data distribution을 갖는다. 











[1]: https://medium.com/neuralspace/bayesian-neural-network-series-post-1-need-for-bayesian-networks-e209e66b70b2
[2]: https://medium.com/neuralspace/bayesian-neural-network-series-post-2-background-knowledge-fdec6ac62d43
[3]: https://arxiv.org/abs/1505.05424
[4]: 
[5]: 
