@setup_workload begin
    symbols1 = [:H, :V, :D, :A, :R, :L]
    symbols2 = [:H, :V, :D, :R]

    @compile_workload begin
        measurement1 = Measurement(get_projector(polarization_state(Val(s))) / 3 for s in symbols1)
        measurement2 = ProportionalMeasurement(get_projector(polarization_state(Val(s))) / 3 for s in symbols2)

        methods = [LinearInversion(), MaximumLikelihood(), BayesianInference()]
        states = [sample(GinibreEnsamble(2)), sample(HaarVector(2))]

        for method ∈ methods, state ∈ states, measurement ∈ [measurement1, measurement2]
            outcomes = simulate_outcomes(state, measurement, 1)
            σ = prediction(outcomes, measurement, method)[1]
            fidelity(state, σ)
        end
    end
end