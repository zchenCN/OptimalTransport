## Optimal Transport Overview

Optimal transportation (OT) problem was first study by Gaspard Monge in 1781: A worker with a shovel in hand want to move a large pile of sand lying on a construction site and wish to minimize her total effort.  OT arouse the interest of mathematicians because it can compare two probability distribution. OT has been rediscovered in many settings and under different forms, giving it a rich history. Kantorovich in 1940s established its significance to logistics and economics. Dantzig solved it numerically in 1949 within the framework of linear programming, giving OT a firm footing in optimization. In recent years, thanks to the emergence of approximate solvers that can scale to large problem dimension, OT is being increasingly used to unlock various problems in imaging sciences (such as color or texture processing), graphics (for shape manipulation) or machine learning (for regression, classification and generative modeling).  

We mainly focus on the numerical aspect of OT, for more theoretical detail, please reference the work of Villani.  

## Monge Problem

We say $\textbf{a}$ is an histogram or probability vector if  it belongs to the probability simplex: 
$$
\Sigma_n = \left\{\textbf{a}\in\mathbb{R}^n_+: \sum_{i=1}^na_i = 1\right\}
$$
That is to say, the elements of $\textbf{a}$ is nonnegative and the sum of them is one. A discrete measure with weights $\textbf{a}$ and locations $x_1, x_2, \cdots, x_n$ reads:
$$
\alpha = \sum_{i=1}^na_i\delta_{x_i}
$$
where $\delta_x$ is the Dirac at  position $x$  intuitively a unit of mass which is infinitely concentrated at location $x$.  For discrete measure:
$$
\alpha = \sum_{i=1}^na_i\delta_{x_i}\quad \text{and}\quad\beta = \sum_{i=1}^mb_i\delta_{y_i}
$$
the Monge problem seeks a map the associates to each point $x_i$ a single point $y_i$ and which must push the mass of $\alpha$ toward the mass of $\beta$, namely, such a map $T:\{x_1, \cdots, x_n\} \rightarrow \{y_1, \cdots, y_m\}$ must verify that:
$$
\forall j \in \{1, 2, \cdots, m\}, b_j = \sum_{T(x_i)=y_j}a_i
$$
which we write in compact form as:
$$
T_{\sharp}\alpha = \beta
$$
Given a cost function $c(x, y)$, the Monge problem is to find the map that minimize the total cost of the transportation:
$$
\min_T\left\{\sum_{i}c(x_i, T(x_i)): T_{\sharp}\alpha=\beta\right\}
$$
Monge maps may not even exist between a discrete measure to another.  

## Kantorovich Relaxation

The key idea of Kantorovich is to relax the deterministic nature of transportation, namely the fact that a source point $x_i$ can only be assigned to another point or location $T(x_i)$ only.  Kantorovich proposed instead that the mass at any point $x_i$ be potentially dispatched across several locations. This flexibility is encoded using a coupling matrix $P \in \mathbb{R}^{n\times m}_+$, where $P_{ij}$ describes the amount of mass flowing from bin $i$ to bin $j$.  Admissible couplings admit a simple characterization that:
$$
U(\textbf{a}, \textbf{b}) = \left\{P\in\mathbb{R}^{n\times m}_+: \sum_jP_{ij}=a_i, \sum_iP_{ij}=b_j\right\}
$$
The set of matrices $U(\textbf{a}, \textbf{b})$ is bounded and defined by $n+m$ equality constraints, and therefore is a convex polytope.  

Given a cost matrix $C$, Kantorovich's OT problem now reads:
$$
L_C(\textbf{a}, \textbf{b}) = \min_{P\in U(\textbf{a}, \textbf{b})}\left<P, C\right> = \sum_{i,j}C_{ij}P_{ij}
$$
This is a linear program and as is usually the case with such programs, its optimal solutions are not necessarily unique.

## Wasserstein Distance

An import feature if OT is that it defines a distance between histograms and probability measures as soon as the cost matrix satisfies certain suitable properties. We suppose $n=m$ and that for some $p\geq 1$, $C = D^p$ where $D\in \mathbb{R}^{n\times n}_+$ is a distance matrix, that is to say $D$ satisfy following properties:

- $D$ is symmetric 

- $D_{ij} = 0$ if and only if $i=j$ 

- $\forall i, j, k, D_{ik} \leq D_{ij} + D_{jk}$   


Then we can define so-called Wasserstein distance on probability simplex $\Sigma_n$, 
$$
W_p(\textbf{a}, \textbf{b}) = L_{D^p}(\textbf{a}, \textbf{b})^{1/p}
$$

## Entropic Regularization

We will introduce a family of numerical schemes to approximate solutions to Kantorovich formulation of OT. It operates by adding an entropic regularization to the original problem. The minimization of the regularized problem can be solved by using a simple alternate minimization scheme which are iterations of simple matrix-vector products. The resulting approximate distance is smooth with respect to input histogram weights and can be differentiated using automatic differentiation.  

The discrete entropy of the coupling matrix is defined as:
$$
H(P) = -\sum_{i, j}P_{ij}(\log(P_{ij})-1)
$$
The idea of the entropic regularization of OT is to use $-H$ as a regularization function to obtain approximate solutions to the origin Kantorovich OT problem:
$$
L^{\epsilon}_C(\textbf{a}, \textbf{b}) = \min_{P\in U(\textbf{a, \textbf{b}})}\left<P, C\right> - \epsilon H(P)
$$
Since the objective is an $\epsilon-strongly$ convex function, the problem mentioned above has a unique optimal solution.  

It has been proved that:
$$
L^{\epsilon}_C(\textbf{a}, \textbf{b})\stackrel{\epsilon\rightarrow0}{\longrightarrow}L_C(\textbf{a}, \textbf{b})
$$


## Sinkhorn's Algorithm

Let $K$ denote the Gibbs kernel associated to the cost matrix $C$ as:
$$
K_{ij} = e^{-\frac{C_{ij}}{\epsilon}}
$$
The solution of the regularized OT problem has the form:
$$
P_{ij} = u_iK_{ij}v_j
$$
for two (unknow) scaling variable $(\textbf{u}, \textbf{v}) \in \mathbb{R}^n_+\times\mathbb{R}^m_+$.  

The factorization of the optimal solution can be conveniently rewritten in matrix form as:
$$
P = diag(\textbf{u})Kdiag(\textbf{v})
$$
The scaling variables must therefore satisfy the following nonlinear equations which correspond to the mass conservation constraints inherent to $U(\textbf{a}, \textbf{b})$: 
$$
\textbf{u}\odot(K\textbf{v}) = \textbf{a}, \quad \textbf{v}\odot(K^T\textbf{u}) = \textbf{b}
$$
where $\odot$ corresponds to entrywise multiplication of vectors. That problem is known as matrix scaling problem which can be solved iteratively by modifying first $\textbf{u}$ so that it satisfies the left-hand side of above equations and then $\textbf{v}$ to satisfy its right-hand side. These two updates define Sinkhorn's algorithm:
$$
\textbf{u}^{(l+1)} = \frac{\textbf{a}}{K\textbf{v}^(l)}, \quad \textbf{v}^{(l+1)} = \frac{\textbf{b}}{K^T\textbf{u}^{(l+1)}}
$$
initialized with an arbitrary positive vector $\textbf{v}^{(0)} = \mathbb{1}_m$. The division operator used above between two vectors is to be understood entrywise.

## References

- [Computational Optimal Transport](https://optimaltransport.github.io/)

