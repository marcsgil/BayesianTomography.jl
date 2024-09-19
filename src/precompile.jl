@setup_workload begin
    symbols = [:H, :V, :D, :A, :R, :L]

    @compile_workload begin
        measurements = [get_projector(polarization_state(Val(s))) for s in symbols]
        problem = StateTomographyProblem(measurements)

        methods = [LinearInversion(problem), MaximumLikelihood(problem), BayesianInference(problem)]
        states = [sample(GinibreEnsamble(2)), sample(HaarVector(2))]

        for method ∈ methods, state ∈ states
            outcomes = simulate_outcomes(state, measurements, 1)
            prediction(outcomes, method)
        end
    end
end