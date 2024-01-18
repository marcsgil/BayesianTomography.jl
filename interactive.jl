### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 58a6c0ce-ae64-11ee-0986-b93b95920a73
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using BayesianTomography, CairoMakie, PlutoUI, LinearAlgebra, Distributions
end

# ╔═╡ 587ef382-b01e-4bc3-a625-0b70c7cc5b87
function formatter(value::Number)
    if value == 0
        return "0"
    elseif value ≈ π
        return L"\pi"
    elseif isinteger(value / π)
        return L"%$(Int(value/π)) \pi"
    else
        return L"%$(value/π) \pi"
    end
end;

# ╔═╡ 9c31d27e-b20a-498e-ad1c-5f33f26ccde9
formatter(values::AbstractVector) = formatter.(values);

# ╔═╡ 3682e1a4-9463-4d16-9224-68ad3543b7c6
order = 1;

# ╔═╡ d8457d46-f43d-480a-9ae2-65e73d0b2498
r = LinRange(-3, 3, 9);

# ╔═╡ 305094c8-4e01-49a5-a42d-f7db1c9d5505
position_operators = assemble_position_operators(r, r, order);

# ╔═╡ 1f73e64b-70fd-4c7e-ad64-72ad73b273cf
mode_converter = diagm([im^k for k ∈ 0:order]);

# ╔═╡ 1c94357d-db1a-4f55-9a0c-011ba9efaf84
operators = hermitianpart.(augment_povm(position_operators, mode_converter));

# ╔═╡ e2877d7d-058f-4aab-b16f-a5a0306cf208
θs = LinRange(0, π, 256);

# ╔═╡ 88e951b6-5d50-4106-84af-c8fa71af0ee8
ϕs = LinRange(0, 2π, 256);

# ╔═╡ 067704f6-db9a-4d93-99db-5c78c8122e3f
true_angles = [π/2, π];

# ╔═╡ b7fb5d16-59dd-4d28-ad0e-c0c3723654fd
ψ = hurwitz_parametrization(true_angles);

# ╔═╡ 557c9fc7-7041-484b-8d08-665bb6027095
begin
	atol=1e-3
	N = 1024
	probs = [real(dot(ψ, E, ψ)) for E in operators]
    @assert minimum(probs) ≥ -atol "The probabilities must be non-negative"
    S = sum(probs)
    @assert isapprox(S, 1; atol) "The sum of the probabilities is not 1, but $S"
    dist = Categorical(map(x -> x > 0 ? x : 0, normalize(vec(probs), 1)))
    samples = rand(dist, N)
end

# ╔═╡ 7c1db365-a212-47c5-b603-ed8d1bf54997
@bind nobs PlutoUI.Slider(1:256;show_value=true)

# ╔═╡ 693a8f56-629f-4d7f-8e48-8a5b37c748ce
begin
	outcomes = Dict{Int,Int}()
	for outcome ∈ samples[1:nobs]
	    outcomes[outcome] = get(outcomes, outcome, 0) + 1
	end
	outcomes
end

# ╔═╡ c3e6a4c7-299c-4821-b403-9a9488e04b46
log_posteriors = [log_likellyhood(outcomes, operators, [θ, ϕ]) + log_prior([θ, ϕ])
                  for θ in θs, ϕ in ϕs];

# ╔═╡ 539fbd17-06bd-4f56-b6b9-eafa539dbdbd
M = minimum(abs, filter(isfinite, log_posteriors));

# ╔═╡ 707a7201-ee20-4382-bcbc-59ff7dc6214e
posteriors = exp.(log_posteriors .+ M);

# ╔═╡ c8e12fb2-fab1-4d44-9130-df5cf82f0d87
normalize!(posteriors, Inf);

# ╔═╡ 98ded703-98e7-4e7d-91be-c5675b893610
pred = prediction(outcomes, operators, MetropolisHastings());

# ╔═╡ 4f8799f6-7921-4a74-bc01-779c39732bd2
begin
	fig = Figure()
	ax = Axis(fig[1, 1],
	    xticks=LinRange(0, π, 5),
	    yticks=LinRange(0, 2π, 5),
	    xtickformat=formatter,
	    ytickformat=formatter,
	    xlabel=L"\theta",
	    ylabel=L"\phi",
		title="$nobs Observations")
	
	
	hm = heatmap!(ax, θs, ϕs, posteriors, colormap=:jet)
	Colorbar(fig[1, 2], hm, label="Rescaled Posterior")
	scatter!(ax, true_angles[1], true_angles[2], color=:yellow, markersize=30, marker=:cross)
	scatter!(ax, pred[1], pred[2], color=:black, markersize=30, marker=:cross)
	fig
end

# ╔═╡ Cell order:
# ╠═58a6c0ce-ae64-11ee-0986-b93b95920a73
# ╠═587ef382-b01e-4bc3-a625-0b70c7cc5b87
# ╠═9c31d27e-b20a-498e-ad1c-5f33f26ccde9
# ╠═3682e1a4-9463-4d16-9224-68ad3543b7c6
# ╠═d8457d46-f43d-480a-9ae2-65e73d0b2498
# ╠═305094c8-4e01-49a5-a42d-f7db1c9d5505
# ╠═1f73e64b-70fd-4c7e-ad64-72ad73b273cf
# ╠═1c94357d-db1a-4f55-9a0c-011ba9efaf84
# ╠═e2877d7d-058f-4aab-b16f-a5a0306cf208
# ╠═88e951b6-5d50-4106-84af-c8fa71af0ee8
# ╠═b7fb5d16-59dd-4d28-ad0e-c0c3723654fd
# ╠═557c9fc7-7041-484b-8d08-665bb6027095
# ╠═693a8f56-629f-4d7f-8e48-8a5b37c748ce
# ╠═c3e6a4c7-299c-4821-b403-9a9488e04b46
# ╠═539fbd17-06bd-4f56-b6b9-eafa539dbdbd
# ╠═707a7201-ee20-4382-bcbc-59ff7dc6214e
# ╠═c8e12fb2-fab1-4d44-9130-df5cf82f0d87
# ╠═98ded703-98e7-4e7d-91be-c5675b893610
# ╠═067704f6-db9a-4d93-99db-5c78c8122e3f
# ╠═7c1db365-a212-47c5-b603-ed8d1bf54997
# ╟─4f8799f6-7921-4a74-bc01-779c39732bd2
