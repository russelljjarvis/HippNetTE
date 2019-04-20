println("reminder to Russell")
println("this is odd, but launch julia using sudo")

println("to install packages use")
println("try to import module first using, 'using'")
println("include('ml.jl')")
#println("Pkg.add('PyCall')")
println("\n\n\n\n\n\n")
ENV["PYTHON"] = "/opt/conda/bin/python"
#=using Knet;
using DataFrames;
#using IJulia
using PyPlot;
using Plots;
using GR;
using Knet;
using Plots;
#using IJulia
using PyPlot;
using Plots;
using GR;
using StatsPlots;
=#
using Conda
using ModernGL
using PyCall;

# using SGDOptim
pushfirst!(PyVector(pyimport("sys")["path"]),"")
pushfirst!(PyVector(pyimport("sys")["path"]),"/opt/conda/bin/python")
pushfirst!(PyVector(pyimport("sys")["path"]),"/home/jovyan/neuronunit")
pushfirst!(PyVector(pyimport("sys")["path"]),"/home/jovyan/work/sciunit")
pushfirst!(PyVector(pyimport("sys")["path"]),"/opt/conda/lib/python3.5/site-packages")
pushfirst!(PyVector(pyimport("sys")["path"]),"/opt/conda/lib/python3.5/site-packages/neuron")
@pyimport pyNN
#@pyimport neurounit

#print(varinfo())
#print(varinfo())

#@pyimport neuronunit
#py"""from neuronunit.optimisation import optimisation_management as om"""
#with open('GA_init_for_julia_objective_raw.p','wb') as f: pickle.dump(ga_out,f)

@pyimport pickle

#binary_trains.p cell_indexs.p cell_names.p #internal_connectivities.p wire_map_online.p

try
    readall(`/opt/conda/bin/python -c "import qi_ascoli"`)
catch
    run(`/opt/conda/bin/python -c "import qi_ascoli"`)
end
f = pybuiltin("open")("binary_trains.p","rb")
p = pickle.Unpickler(f)
bts = p[:load]()
f[:close]()
f = pybuiltin("open")("cell_names.p","rb")
p = pickle.Unpickler(f)
bts = p[:load]()
f[:close]()
f = pybuiltin("open")("cell_indexs.p","rb")
p = pickle.Unpickler(f)
bts = p[:load]()
f[:close]()
#for c in ts:
#    println(c[1][1].scores)

print(bts)

#=
@pyimport pickle
f = pybuiltin("open")("pipe_tests.p","rb")
p = pickle.Unpickler(f)
pipe_tests = p[:load]()
print(pipe_tests)
f[:close]()
println("semi colons just suppress output in julia.")
py"""
import pickle
with open("pipe_tests.p","rb") as f:
    test_frame = pickle.load(f)

for k in keys(test_frame)
    println(k, " ==> ", test_frame[k])
    use_test = test_frame[k]

#for key, use_test in test_frame.items():
    # use the best parameters found via the sparse grid search above, to inform the first generation
    # of the GA.
    if str('results') in MODEL_PARAMS['RAW'].keys():
        MODEL_PARAMS['RAW'].pop('results', None)

    #=
    Just want to initialize the GA and not to use it.
    _, Opt = om.run_ga(MODEL_PARAMS['RAW'], 0, use_test, free_params = MODEL_PARAMS['RAW'],
                                NSGA = True, MU = MU, model_type = str('RAW'),seed_pop=seeds[key])
    =#
"""
=#



#catch
#    println("pass")
#end
#scraped_new[1]["wcount"]
#sn = [ sn["wcount"] for sn in scraped_new ]
#varinfo(sn)
#gr()
#histogram(sn)
#using PyPlot
#h = PyPlot.plt.hist(sn)
#png("document_length_distribution.png")
#varinfo()

include(Knet.dir("data","housing.jl"))
x,y = housing()

predict(w, x) = w[1] * x .+ w[2]
loss(w, x, y) = mean(abs2, predict(w, x)-y)
lossgradient = grad(loss)

function train(w, data, lr=0.01; ncores::Int=8)
    #@parallel (+)
    for (x,y) in data
        dw = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] -= dw[i]*lr
        end
    end
    return w
end;

#plotly();
#gr();
#scatter(x', y[1,:], layout=(3,5), reg=true, size=(950,500))
#png("dnn_scatter_plot.png")
#BigInt cntr;

println("try to import module first using, 'using'")
#println("Pkg.add('PyCall')")
println("\n\n\n\n\n\n")

function parallel_net_computation(N::Int; ncores::Int=8)
   #@parallel (+)
   w = Any[ 0.1*randn(1,13), 0.0 ];
   errdf = DataFrame(Epoch=1:20, Error=0.0);

   for i=1:N
    cntr = 1
    #println("cntr",cntr)

      #println(i,cntr)
      w = train(w, [(x,y)])
      if mod(i, 10) == 0
          println("Epoch $i: $(round(loss(w,x,y)))")
          errdf[cntr, :Epoch]=i
          errdf[cntr, :Error]=loss(w,x,y)
          print(errdf[cntr, :Error])
          cntr+=1
      end
  end;
  return errdf
end
errdf = parallel_net_computation(200)
print(errdf)




function parallel_pi_computation(N::Int; ncores::Int=8)
    #=
    Compute pi in parallel, over ncores cores, with a Monte Carlo simulation throwing N total darts
    =#

    # compute sum of pi's estimated among all cores in parallel
    sum_of_pis = @parallel (+) for i=1:ncores
        compute_pi(ceil(Int, N / ncores))
    end

    return sum_of_pis / ncores  # average value
end
@time parallel_pi_computation(N::Int; ncores::Int=8)
