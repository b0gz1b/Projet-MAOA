using JuMP
using CPLEX
using CPUTime

include("../TSP/TSP_IO.jl")
include("ringStar_compact.jl")
include("ringStar_ncompact.jl")

function Resoud_ringStar(filename)


	I = Read_undirected_TSP(filename)
	println(I)
	
	filename_inst = replace(filename, ".tsp" => "_inst")
    WritePdf_visualization_TSP(I, filename_inst)

	p = 6

	@time @CPUtime stations, affectations, S, time = ring_star_compact(I, p) # on le résout

	println(typeof(affectations))

	val_RingStarCompact = Cout(I, S, affectations)
	mean_RingStarCompact = Mean_cout_dist_min(I, S, affectations)
	ratio_RingStarCompact = Ratio(I, S, affectations)

	println("Solution Ring Star compact :S=",S)
	println("Cout: ",val_RingStarCompact)
	println("Temps moyen: ",mean_RingStarCompact)
	println("Ratio: ",ratio_RingStarCompact)

	println()

	filename_RingStar = replace(filename, ".tsp" => "_RingStar_")

	WritePdf_visualization_solution_ordre(I,S,affectations,filename_RingStar) # on fait le pdf

	@time @CPUtime stations, affectations, S, time = ring_star_ncompact(I, p) # on le résout
 
	val_RingStarNCompact=Compute_value_cycleRingStar(I, S)
	mean_RingStarNCompact = Mean_cout_dist_min(I, S, affectations)
	ratio_RingStarNCompact = Ratio(I, S, affectations)

	println("Solution Ring Star ncompact :S=",S)
	println("Cout: ",val_RingStarNCompact)
	println("Temps moyen: ",mean_RingStarNCompact)
	println("Ratio: ",ratio_RingStarNCompact)

	println()
	
	filename_RingStar = replace(filename, ".tsp" => "_nRingStar_")

	WritePdf_visualization_solution_ordre(I,S,affectations,filename_RingStar) # on fait le pdf



end
		
	

