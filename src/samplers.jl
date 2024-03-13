function sample_haar_unitary(n)
    # Generate a complex random matrix
    Z = randn(ComplexF64, n, n)

    # Perform QR decomposition
    Q, R = qr(Z)

    # Make R's diagonal real and non-negative
    Lambda = Diagonal(sign.(diag(R)))

    # Return the unitary matrix
    return Q * Lambda
end

function sample_haar_vector(n)
    sample_haar_unitary(n)[:, 1]
end

function sample_ginibri_state(d)
    X = rand(ComplexF32, d, d)
    ρ = X * X'
    ρ ./= tr(ρ)
    ρ
end