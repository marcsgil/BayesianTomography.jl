# Theory

$
\def\<{\langle}  %% overiding the original command \<
\def\>{\rangle}  %% overiding the original command \>
\newcommand{\dket}[1]{| #1\>\!\>}
\newcommand{\Dket}[1]{\Bigl| #1\Bigr\>\!\Bigr\>}
\newcommand{\dbra}[1]{\<\!\< #1|}
\newcommand{\Dbra}[1]{\Bigl\<\!\Bigl\< #1\Bigr|}
\newcommand{\dinner}[2]{\<\!\< #1| #2\>\!\>}
\newcommand{\Dinner}[2]{\Bigl\<\!\Bigl\< #1\Bigl| #2\Bigr\>\!\Bigr\>}
\newcommand{\douter}[2]{| #1\>\!\>\<\!\< #2|}
\newcommand{\Douter}[2]{\Bigl| #1\Bigr\>\!\Bigr\>\Bigl\<\!\Bigl\< #2\Bigr|}
\newcommand{\sif}{\mathcal{L}^2(\mathbb{R}^n; \mathbb{C})}
\newcommand{\siftwo}{\mathcal{L}^2(\mathbb{R}^2; \mathbb{C})}
$

## POVMs

Let $\text{Her}(\mathcal{H}) \subset \mathcal{H}$ be the set of hermitian operators acting on a Hilbert space $\mathcal{H}$. Elements of $\text{Her}(\mathcal{H})$ may be written as double kets: $\dket{F}$. Such a space is also a *real* Hilbert space under the inner product
$$
    \dinner{E}{F} = \text{Tr} E^\dagger F = \text{Tr} E F .
$$
The state of a quantum system is represented by an element $\rho$ of the set of positive semi-definite operators $\text{Pos}(\mathcal{H}) \subset \text{Her}(\mathcal{H})$ such that $\text{Tr} \rho = 1$. Observables are represented by elements $A \in \text{Her}(\mathcal{H})$ so that its expectation value is given by $\langle A \rangle = \dinner{A}{\rho}$. A Positive Operator Valued Measure (POVM) is a set of observables $\{F_m\} \subset \text{Pos}(\mathcal{H})$ with the property that $\sum_m F_m = I$, where $I$ is the identity operator. These operators model the possible outcomes of an experiment: outcome $m$ happens with probability $p(m) = \dinner{F_m}{\rho}$. The POVM conditions ensure that $p(m) \ge 0$ and $\sum_m p(m) = 1$.

## Linear Inversion

The simplest method which solves the problem of quantum state tomography is linear inversion, which we describe in what follows. A POVM with $M$ elements induces a linear map $T: \text{Her}(\mathcal{H}) \to \mathbb{R}^M$ defined by the expression
$$
T\dket{\Omega} = \left(\dinner{F_1}{\Omega},\ldots,\dinner{F_M}{\Omega}\right).
$$
In order to be suitable for tomography, we want that the measurement probabilities $\mathbf{p} = T \dket{\rho}$ uniquely determine the state $\dket{\rho}$. This is assured if the transformation $T$ is injective. A POVM with this property is said to be informationally complete.

By choosing a basis $\{\Omega_n\} \subset \text{Her}(\mathcal{H})$, we can specify an arbitrary state $\dket{\rho}$ by a list of coefficients $\mathbf{x} = (x_1,\ldots,x_N)$ such that $\dket{\rho} = \sum_n x_n \dket{\Omega_n}$. Then, denoting by $\mathbb{T}$ the matrix of $T$ with respect to the canonical basis of $\mathbb{R}^M$ and $\{\Omega_m\}$, which has entries $\mathbb{T}_{mn} = \dinner{F_m}{\Omega_n}$, we have $\mathbf{p} = \mathbb{T}\mathbf{x}$. When $\mathbb{T}$ is injective, $\mathbb{T}^\dagger\mathbb{T}$ is invertible, and then we can explicitly write the solution of the above equation as
$$
\mathbf{x} = (\mathbb{T}^\dagger\mathbb{T})^{-1} \mathbb{T}^\dagger \mathbf{p},
$$
which, in theory, solves the tomography problem.

## Bayesian mean inference

## Computational representation