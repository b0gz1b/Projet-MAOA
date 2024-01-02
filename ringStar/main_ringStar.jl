using JuMP
using CPLEX
using CPUTime

include("../TSP/TSP_IO.jl")
include("compact.jl")

function Resoud_ringStar(filename)
	
	I = Read_undirected_TSP(filename)
	println(I)
	
	filename_inst = replace(filename, ".tsp" => "_inst")
    WritePdf_visualization_TSP(I, filename_inst)

	@time @CPUtime stations, affectations, cycle =ring_star_compact(I, 5) # on le résout
 
	val_RingStarCompact=Compute_value_RingStar(I, cycle)
	println("Solution Ring Star compact :S=",cycle)
	println("Valeur: ",val_RingStarCompact)
	println()
	
	filename_RingStar = replace(filename, ".tsp" => "_RingStar")

	WritePdf_visualization_solution_ordre(I,cycle,filename_RingStar) # on fait le pdf


end
		
	

