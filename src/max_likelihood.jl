struct MaximumLikelihood{T}
    problem::StateTomographyProblem{T}
end

function log_likelihood!(buffer, frequencies, traceless_part, trace_part, θ)
    get_probabilities!(buffer, traceless_part, trace_part, θ)
    broadcast!(log, buffer, buffer)
    frequencies ⋅ buffer
end

function update_θ!(θ, ρ, t, ∇ℓπ)
    @. θ = θ + t * ∇ℓπ
    density_matrix_reconstruction!(ρ, θ)
    project2density!(ρ)
    gell_mann_projection!(θ, ρ)

end

function gradient_ascent!(θ, θ_candidate, buffer1, buffer2, ∇ℓπ, ρ, δ, frequencies, trace_part, traceless_part, t, γ, max_iter, tol)
    for i in 1:max_iter
        ℓ = BayesianTomography.log_likelihood!(∇ℓπ, buffer1, buffer2,
            frequencies, traceless_part, trace_part, θ)

        ti = t
        update_θ!(θ_candidate, ρ, ti, ∇ℓπ)
        @. δ = θ_candidate - θ

        # Backtracking line search
        while log_likelihood!(buffer1, frequencies, traceless_part, trace_part, θ_candidate) ≤ ℓ + real(∇ℓπ ⋅ δ - δ ⋅ δ / (2ti))
            ti *= γ
            update_θ!(θ_candidate, ρ, ti, ∇ℓπ)
            @. δ = θ_candidate - θ
        end

        if sum(abs2, δ) < tol
            break
        end

        copy!(θ, θ_candidate)
    end
end

function prediction(outcomes, method::MaximumLikelihood{T};
    θ₀=zeros(T, size(method.problem.traceless_part, 2)),
    t=0.1,
    γ=0.5,
    max_iter=10^5,
    tol=1e-12) where {T}

    I = findall(!iszero, vec(outcomes))
    frequencies = normalize(outcomes, 1)[I]
    traceless_part = method.problem.traceless_part[I, :]
    trace_part = method.problem.trace_part[I]
    buffer1 = similar(trace_part)
    buffer2 = similar(trace_part)
    ∇ℓπ = similar(θ₀)
    θ = copy(θ₀)
    θ_candidate = similar(θ)
    ρ = density_matrix_reconstruction(θ)
    δ = similar(θ)

    gradient_ascent!(θ, θ_candidate, buffer1, buffer2, ∇ℓπ, ρ, δ, frequencies, trace_part, traceless_part, t, γ, max_iter, tol)

    density_matrix_reconstruction!(ρ, θ)
    post_measurement_state!(ρ, method.problem.inv_kraus_operator)
    gell_mann_projection!(θ, ρ)

    ρ, θ
end