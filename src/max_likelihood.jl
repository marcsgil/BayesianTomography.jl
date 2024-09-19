struct MaximumLikelihood{T1, T2}
    problem::StateTomographyProblem{T1, T2}
end

function log_likelihood!(∇ℓπ, buffer1, buffer2, frequencies, traceless_povm, correction, y, x)
    get_probabilities!(buffer1, traceless_povm, correction, y)

    if any(x -> x < 0, buffer1)
        y .= x
        get_probabilities!(buffer1, traceless_povm, correction, y)
    end

    broadcast!(/, buffer2, frequencies, buffer1)
    mul!(∇ℓπ, traceless_povm', buffer2)
    broadcast!(log, buffer1, buffer1)
    frequencies ⋅ buffer1
end


function log_likelihood!(buffer, frequencies, traceless_part, trace_part, x)
    get_probabilities!(buffer, traceless_part, trace_part, x)
    broadcast!(log, buffer, buffer)
    frequencies ⋅ buffer
end

function update_x!(x, y, ρ, t, ∇ℓπ)
    @. x = y + t * ∇ℓπ
    density_matrix_reconstruction!(ρ, x)
    project2density!(ρ)
    gell_mann_projection!(x, ρ)
end



function gradient_ascent!(x, x_prev, y, buffer1, buffer2, ∇ℓπ, ρ, δ, δ_hat, frequencies, trace_part, traceless_part, t, β, max_iter, tol)
    θ = 1

    for i in 1:max_iter
        ℓ = log_likelihood!(∇ℓπ, buffer1, buffer2,
            frequencies, traceless_part, trace_part, y, x)

        update_x!(x, y, ρ, t, ∇ℓπ)
        @. δ = x - y

        # Backtracking line search
        while log_likelihood!(buffer1, frequencies, traceless_part, trace_part, x) ≤ ℓ + real(∇ℓπ ⋅ δ - (δ ⋅ δ) / (2t))
            t *= β
            update_x!(x, y, ρ, t, ∇ℓπ)
            @. δ = x - y
            iszero(δ) && return nothing
        end

        @. δ_hat = x - x_prev

        if sum(abs2, δ) < tol
            break
        end

        if δ ⋅ δ_hat < 0
            θ = 1
            x .= x_prev
            y .= x_prev
        else
            new_θ = (1 + sqrt(1 + 4 * θ^2)) / 2
            @. y = x + (θ - 1) * δ_hat / new_θ

            x_prev .= x
            θ = new_θ
        end
    end
end

function prediction(outcomes, method::MaximumLikelihood{T1, T2};
    x₀=zeros(T1, size(method.problem.traceless_part, 2)),
    t=0.4,
    β=0.8,
    max_iter=10^3,
    tol=1e-10) where {T1, T2}

    I = findall(!iszero, vec(outcomes))
    frequencies = normalize(outcomes[I], 1)
    traceless_part = method.problem.traceless_part[I, :]
    trace_part = method.problem.trace_part[I]
    buffer1 = similar(trace_part)
    buffer2 = similar(trace_part)
    ∇ℓπ = similar(x₀)
    x = copy(x₀)
    x_prev = copy(x)
    y = copy(x)
    ρ = density_matrix_reconstruction(x)
    δ = similar(x)
    δ_hat = similar(x)

    gradient_ascent!(x, x_prev, y, buffer1, buffer2, ∇ℓπ, ρ, δ, δ_hat, frequencies, trace_part, traceless_part, t, β, max_iter, tol)

    density_matrix_reconstruction!(ρ, y)
    post_measurement_state!(ρ, method.problem.inv_kraus_operator)
    gell_mann_projection!(y, ρ)

    ρ, y
end