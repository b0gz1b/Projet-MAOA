using MathOptInterface
using JuMP
using CPLEX

include("../TSP/TSP_IO.jl")

function ring_star_compact(G, p)

    c = calcul_dist(G)
    
	println("Création du PLNE")
    m = Model(CPLEX.Optimizer)

    @variable(m, y[1:G.nb_points, 1:G.nb_points], Bin)

    @variable(m, x[1:G.nb_points, 1:G.nb_points] , Bin)

    @variable(m, 0 <= z[1:G.nb_points, 1:G.nb_points] <= p-1, Int)

    @objective(m, Min, sum(sum(c[i, j] * x[i, j] for j = 1:G.nb_points) for i = 1:G.nb_points) + sum(sum(c[i, j] * y[i, j] for j = 1:G.nb_points) for i = 1:G.nb_points))

    #pas d'arete i à i
    for i in 1:G.nb_points
        @constraint(m, x[i, i] == 0)
    end

    #contraintes de stations
    @constraint(m, y[1, 1] == 1)

    for j in 2:G.nb_points
		@constraint(m, y[1, j] == 0)
	end

    @constraint(m, sum(y[i, i] for i in 1:G.nb_points) == p) #(1)

    for i in 1:G.nb_points
        @constraint(m, sum(y[i, j] for j in 1:G.nb_points) == 1) #(2)
    end

    for i in 1:G.nb_points
        for j in 1:G.nb_points
            if i!=j
                @constraint(m, y[i, j] <= y[j, j]) #(3)
            end
        end
    end

    for i in 1:G.nb_points
        @constraint(m, sum(x[j, i] for j in 1:i-1) + sum(x[i, j] for j in i+1:G.nb_points) == 2 * y[i, i]) #(4)
    end

    #contraintes de flot
    @constraint(m, sum(z[1, j] for j in 2:G.nb_points) == p - 1) #(5)
    
    #z[i,j] def pour tout i et pour tout j\{1,i}
    for i in 1:G.nb_points
        @constraint(m, z[i, i] == 0)
        @constraint(m, z[i, 1] == 0)
    end

    for i in 2:G.nb_points
        @constraint(m, sum(z[j, i] for j in 1:i-1) + sum(z[j, i] for j in i+1:G.nb_points) == sum(z[i, j] for j in 2:i-1) + sum(z[i, j] for j in i+1:G.nb_points) + y[i, i]) #6
    end

    for i in 1:G.nb_points
        for j in 2:G.nb_points
            if j!=i
                @constraint(m, z[i, j] + z[j, i] <= (p - 1) * x[i, j]) #7
            end
        end
    end

    # for j in 2:G.nb_points
    #   # on veut i dans V \ {j}
    #   @constraint(m, z[1, j] <= (p-1)*x[1, j])
    # end

    # print(m)
	println()
	
	println("Résolution du PLNE par le solveur")
	optimize!(m)
   	println("Fin de la résolution du PLNE par le solveur")
   	
	#println(solution_summary(m, verbose=true))

	status = termination_status(m)
 
	# un petit affichage sympathique
	if status == MathOptInterface.OPTIMAL
		println("Valeur optimale = ", objective_value(m))
		println("Solution primale optimale :")
		
        stations = Int64[]
        for i in 1:G.nb_points
            if (value(y[i, i]) > 0.999)
                push!(stations, i)
            end
        end

        println("Stations = ", stations)

        affectations = []
        for station in stations
            l = Int64[]
            push!(l, station)
            for i in 1:G.nb_points
                if i!=station
                    if (value(y[i, station]) > 0.999)
                        push!(l, i)
                    end
                end
            end
            push!(affectations, l)
        end

        println("Affectations =", affectations)

		cycle = Int64[]
        i=1
        j=2
        while (value(x[i, j]) < 0.999) 
     	    j=j+1
        end        
        push!(cycle,1)
        push!(cycle,j)
        i=j
        tmp=1
        while (i!=1)
            j=1
            while  ( j==i || (value(x[i,j]) < 0.999 && value(x[j,i]) < 0.999) || j==tmp ) 
                j=j+1
            end
            push!(cycle,j)
            tmp=i
            i=j
		end
        println("S = ", cycle)
        println("Temps de résolution :", solve_time(m))

        return stations, affectations, cycle
	else
		 println("Problème lors de la résolution")
	end
end
