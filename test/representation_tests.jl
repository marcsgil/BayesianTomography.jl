function random_dict(length, max)
    return Dict(n => rand(1:max) for n in 1:length)
end

let dict = random_dict(10, 10)
    @test array2dict(dict2array(dict, 10)) == dict
end

let dict = random_dict(100, 10)
    @test array2dict(dict2array(dict, (10, 10))) == dict
end

let dict = random_dict(1000, 10)
    @test array2dict(dict2array(dict, (10, 10, 10))) == dict
end

let array = rand(1:100, 10)
    @test dict2array(array2dict(array), 10) == array
end

let array = rand(1:100, (10, 10))
    @test dict2array(array2dict(array), (10, 10)) == array
end

let array = rand(1:100, (10, 10, 10))
    @test dict2array(array2dict(array), (10, 10, 10)) == array
end

let history = rand(1:10, 100)
    @test history2array(history, 100) == dict2array(history2dict(history), 100)
    @test history2dict(history) == array2dict(history2array(history, 100))
end