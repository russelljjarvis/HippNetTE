
using Pkg;
ENV["PYTHON"] = "/opt/conda/bin/python"
#try
#   Pkg.rm("PyCall")
#catch
Pkg.add("PyCall")

#end
Pkg.build("PyCall")

Pkg.add("Conda")

using Conda
Conda.add("beautifulsoup4")
Conda.add("nbformat")
Pkg.add("ModernGL")
ENV["MODERNGL_DEBUGGING"] = "true";
Pkg.build("ModernGL")
Pkg.add("PyPlot")
Pkg.add("Conda")
Pkg.add("ArrayFire")
Pkg.add("StatsPlots")
Pkg.add("PyCall")
Pkg.add("GR")
Pkg.add("Plots");
Pkg.add("StatsPlots"); #to install the StatPlots package.
Pkg.add("DataFrames");
Pkg.add("Seaborn")
Pkg.add("PyPlot")
Pkg.add("Knet")
# Pkg.add("SGDOptim")
