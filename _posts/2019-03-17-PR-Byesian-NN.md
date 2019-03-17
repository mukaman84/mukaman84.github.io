본 포스트는 다음의 post를 간단히 요약한 내요입니다.

https://medium.com/@joeDiHare/deep-bayesian-neural-networks-952763a9537

글쓴이는 neural network의 불확실성을 모델링하기 위해 3개의 bayesian approach를 제안한다.

1) MCMC integral의 근사화
2) black-box variational inference
3) MC dropout사용

기존의 전통적인 방식은 likekihood 최대로하는 optimal value를 학습하는, 즉 weight와 bias를 예측하는 방법이었다. 이 경우 weight와 bias는 scalars 값을 가진다. 반면에, Bayesian approach는 이 파라미터들의 분포에 관한 것이다. 

예를들어 앞선 weight와 bias는 학습이 완료된 네트워크가 가질수 있는 값드의 분포로 표현될 수 있다.
나의 해석으로는 네트워크가 학습될 수 있는 local minima가 엄청나게 많은데 그 값들이 분포를 가짐을 의미한다고 생각한다.

그렇다면 여기서 생각해볼 내용은, single value가 아닌 분포를 가진다는 것이 어떠한 장점이 있을까? 글쓴이는 이를 당연하게 분포의 관점에서 설명한다.
즉, 분포의 의미인 샘플링을 거듭하여 학습하였을 경우에 지속적으로 같은 예측을 한다면 그 결과는 신뢰할만 하다는 내용이다.


![by_curve](https://cdn-images-1.medium.com/max/800/1*zCgkD5l7Tyzrch13ndWOfg.png)



