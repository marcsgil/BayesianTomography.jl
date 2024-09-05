X = rand(ComplexF32, 1000,1000)

ρ1 = X * X'
ρ2 = Hermitian(X * X')

@benchmark eigen($ρ1)
@benchmark eigen($ρ2)

##

@benchmark eigen($ρ1)

ρ1

ρ1
@code_warntype eigen(ρ2, sortby=x->-x)