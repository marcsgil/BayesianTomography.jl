for size ∈ [ntuple(x -> 10, k) for k ∈ 1:4]
    let observations = rand(0:2, size)
        @test (observations
               ==
               complete_representation(reduced_representation(observations), size))
    end

    let history = History(rand(1:prod(size), 10))
        @test (complete_representation(history, size)
               ==
               complete_representation(reduced_representation(history), size))
    end
end