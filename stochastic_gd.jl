
ENV["PYTHON"] = "/opt/conda/bin/python"

using Conda
using ModernGL
using PyCall;

pushfirst!(PyVector(pyimport("sys")["path"]),"")
pushfirst!(PyVector(pyimport("sys")["path"]),"/opt/conda/bin/python")
pushfirst!(PyVector(pyimport("sys")["path"]),"/home/jovyan/neuronunit")
pushfirst!(PyVector(pyimport("sys")["path"]),"/home/jovyan/work/sciunit")
pushfirst!(PyVector(pyimport("sys")["path"]),"/opt/conda/lib/python3.5/site-packages")
pushfirst!(PyVector(pyimport("sys")["path"]),"/opt/conda/lib/python3.5/site-packages/neuron")
@pyimport pickle


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




include(Knet.dir("data","housing.jl"))
x,y = housing()

predict(w, x) = w[1] * x .+ w[2]
loss(w, x, y) = mean(abs2, predict(w, x)-y)
lossgradient = grad(loss)

function train(w, data, lr=0.01; ncores::Int=8)
    @parallel (+)
    for (x,y) in data
        dw = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] -= dw[i]*lr
        end
    end
    return w
end;

gr();
scatter(x', y[1,:], layout=(3,5), reg=true, size=(950,500))
png("dnn_scatter_plot.png")
#BigInt cntr;


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
