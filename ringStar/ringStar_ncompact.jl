using MathOptInterface
using JuMP
using CPLEX
using CPUTime
using Graphs
# using LightGraphs
# using MetaGraphs
# using GraphIO
# using SimpleGraphs
using Combinatorics

include("../TSP/TSP_IO.jl")

# MathOptInterface is a shortcut for MathematicalOptimizationInterface

# Fonction d'algorithme de coupe minimale (Karger's algorithm)
# function mincut(graph)
#     while nv(graph) > 2
#         # Choisissez une arête au hasard
#         edge = rand(1:ne(graph))

#         # Obtenez les extrémités de l'arête
#         src, dst = src(ei(graph, edge)), dst(ei(graph, edge))

#         # Fusionnez les nœuds correspondants
#         contract!(graph, src, dst)
#     end

#     # La coupe résultante est donnée par les deux nœuds restants
#     Part = connected_components(graph)

#     # La valeur de la coupe est le nombre d'arêtes entre les deux partitions
#     valuecut = count(e -> in_component(e.src, Part) != in_component(e.dst, Part), edges(graph))

#     return Part, valuecut
# end

function edge_value(x,i,j)
   if (i<j)
      return value(x[i,j])
   else
      return value(x[j,i])
   end
end

# Si x est entier et respecte les contraintes de degre sum_{v=1}^n x[u,v)]=1 pour tout u
# Renvoie la liste des sommets formant un cycle passant par u
function find_cycle_in_integer_x(x, u)
      S = Int64[]
      #push!(S,u)
      i=u
      prev=-1
      while true
         j=1
         while  ( j==i || j==prev || edge_value(x,i,j) < 0.999 ) 
            j=j+1
         end
         push!(S,j)
         prev=i
         i=j
	 	 #println(i)
         (i!=u) || break   # si i==u alors fin de boucle
      end
      return S
end



function ring_star_ncompact(G, p)

    c = calcul_dist(G)
    
	println("Création du PLNE")
    m = Model(CPLEX.Optimizer)
  
    # Setting some stat variables
    nbViolatedMengerCut_fromIntegerSep = 0
    nbViolatedMengerCut_fromFractionalSep = 0

    @variable(m, y[1:G.nb_points, 1:G.nb_points], Bin)

    @variable(m, x[1:G.nb_points, 1:G.nb_points] , Bin)

    @objective(m, Min, sum(sum(c[i, j] * x[i, j] for j in 1:G.nb_points) for i in 1:G.nb_points) + sum(sum(c[i, j] * y[i, j] for j in 1:G.nb_points) for i in 1:G.nb_points))

    #pas d'arete i à i
    for i in 1:G.nb_points
        @constraint(m, x[i, i] == 0)
    end

    #contraintes de stations
    @constraint(m, y[1, 1] == 1)

    for j in 2:G.nb_points
		@constraint(m, y[1, j] == 0)
	end

    @constraint(m, sum(y[i, i] for i in 1:G.nb_points) == p) #1

    for i in 1:G.nb_points
        @constraint(m, sum(y[i, j] for j in 1:G.nb_points) == 1) #2
    end

    for i in 1:G.nb_points
        for j in 1:G.nb_points
            if i!=j
                @constraint(m, y[i, j] <= y[j, j]) #(3)
            end
        end
    end

    for i in 1:G.nb_points
        @constraint(m, sum(x[j, i] for j in 1:i-1) + sum(x[i, j] for j in i+1:G.nb_points) == 2*y[i, i]) #4
    end

    # for S in sous_ensembles([i for i in 2:G.nb_points], 2)
    #     V_prive_S = setdiff([i for i in 1:G.nb_points], S)
    #     deltaS = couples_possibles(S, V_prive_S)
    #     @constraint(m, sum(sum(x[i, j] for (i,j) in e) for e in sous_ensembles(deltaS, 1)) >= 2 * sum(sum(y[i, i] for j in S) for i in S))
    # end

    for i in 2:G.nb_points
        for j in 1:G.nb_points
            @constraint(m, y[j, j] >= x[i, j]) #9
        end
    end

    # for W in sous_ensembles([i for i in 2:G.nb_points], 2)
    #     V_prive_W = setdiff([i for i in 1:G.nb_points], W)
    #     deltaW = couples_possibles(W, V_prive_W)
    #     @constraint(m, sum(sum(x[i, j] for (i,j) in e) for e in sous_ensembles(deltaW, 1)) >= 2 * sum(sum(y[i, j] for j in W) for i in W))
    # end

   
   # Initialization of a graph to compute min cut for the fractional separation
   
  G_sep=complete_digraph(G.nb_points)

  #################
  # our function lazySep_ViolatedMengerCut
    function lazySep_ViolatedMengerCut(cb_data)
        # cb_data is the CPLEX value of our variables for the separation algorithm
        # In the case of a LazyCst, the value is integer, but sometimes, it is more 0.99999 than 1

        #repérer les p points médiants
        pointsMedians = Int64[]
        for i in 1:G.nb_points
            if (callback_value(cb_data, y[i,i]) > 0.999)
                push!(pointsMedians, i)
            end
        end

        nb_p = size(pointsMedians,1)
        # Get the x value from cb_data and round it
        xsep =zeros(Float64, nb_p, nb_p);
        
        for i in 1:nb_p
           for j in 1:nb_p
                if i!=j
                    if pointsMedians[j] < pointsMedians[i]
                        xsep[i,j]=callback_value(cb_data, x[pointsMedians[j],pointsMedians[i]])
                    else
                        xsep[i,j]=callback_value(cb_data, x[pointsMedians[i],pointsMedians[j]])
                    end
                end
           end
        end
        # for i in 1:G.nb_points
        #   print(xsep[i]," ")
        # end
        # println()
        
#        violated, W = ViolatedMengerCut_IntegerSeparation(G,xsep)
        
        start=rand(1:nb_p)
  
        W = find_cycle_in_integer_x(xsep, start)

        tmp = Int64[]
        for ind in W
            push!(tmp, pointsMedians[ind])
        end
        W = tmp
        #on verifie si le cycle est unique
        # unique = true
        # if 1 not in W
        #     unique = false
        # end
        
        # if unique
        #     for pt in pointsMedians
        #         if pt in W
        #             continue
        #         else
        #             unique = false
        #         end
        #     end
        # end

        #ajout de contrainte si le cycle ne passe pas par 1 ou/et non unique
        if (size(W,1)!=nb_p || !(1 in W))    # size(W) renvoie sinon (taille,)
      
           #println(W)
           for k in W
            con = @build_constraint(sum(x[i,j] for i ∈ W for j ∈ i+1:G.nb_points if j ∉ W) 
                                    + sum(x[j,i] for i ∈ W for j ∈ 1:i-1 if j ∉ W)  
                                    >= 2 * y[k, k])

            MathOptInterface.submit(m, MathOptInterface.LazyConstraint(cb_data), con)
           end
            
          #println(con)
           
           nbViolatedMengerCut_fromIntegerSep=nbViolatedMengerCut_fromIntegerSep+1
           
        end
        
    end
  #
  #################


  #################
  # our function userSep_ViolatedMengerCut
    function userSep_ViolatedMengerCut(cb_data)
        # cb_data is the CPLEX value of our variables for the separation algorithm
        # In the case of a usercut, the value is fractional or integer (and can be -0.001)

        #repérer les p points médiants
        pointsMedians = Int64[]
        for i in 1:G.nb_points
            if (callback_value(cb_data, y[i,i]) > 0.001)
                push!(pointsMedians, i)
            end
        end

        nb_p = size(pointsMedians,1)

        xsep =zeros(Float64, nb_p, nb_p);
        for i in 1:nb_p
            for j in 1:nb_p
                 if i!=j
                     if pointsMedians[j] < pointsMedians[i]
                         xsep[i,j]=callback_value(cb_data, x[pointsMedians[j],pointsMedians[i]])
                     else
                         xsep[i,j]=callback_value(cb_data, x[pointsMedians[i],pointsMedians[j]])
                     end
                 end
                 if xsep[i,j] < 0
                    xsep[i,j] = 0
                 end
            end
         end


       G_sepP = complete_graph(nb_p)
    #    println(is_directed(G_sepP))
    #    println(size(G_sepP))
    #    println(G_sepP)
    #    println(xsep)



       Part,valuecut = mincut(G_sepP, xsep)  # Part is a vector indicating 1 and 2 for each node to be in partition 1 or 2

       W=Int64[]
       for i in 1:nb_p
          if Part[i]==1
             push!(W,pointsMedians[i])
          end
       end
       
        if 1 in W
            W = setdiff(pointsMedians, W)
        end

        
       e = 0.001
       if (valuecut<2.0-e)
      #     println(W)
           
        for k in W
            con = @build_constraint(sum(x[i,j] for i ∈ W for j ∈ i+1:G.nb_points if j ∉ W) 
                                   + sum(x[j,i] for i ∈ W for j ∈ 1:i-1 if j ∉ W)  
                                   >= 2 * y[k, k])

            MathOptInterface.submit(m, MathOptInterface.UserCut(cb_data), con)
        end
           
      #     println(con)
           
            
           nbViolatedMengerCut_fromFractionalSep=nbViolatedMengerCut_fromFractionalSep+1
         
       end
             
    end
  #
  #################

  #################
  # our function primalHeuristicTSP
    function primalHeuristicTSP(cb_data)
    
    pointsMedians = Int64[]
    for i in 1:G.nb_points
        if (callback_value(cb_data, y[i,i]) > 0)
            push!(pointsMedians, i)
        end
    end

     nb_p = size(pointsMedians,1)
     # Get the x value from cb_data 
     xfrac =zeros(Float64, nb_p, nb_p);    

    for i in 1:nb_p
        for j in 1:nb_p
            if i!=j
                if pointsMedians[j] < pointsMedians[i]
                    xfrac[i,j]=callback_value(cb_data, x[pointsMedians[j],pointsMedians[i]])
                else
                    xfrac[i,j]=callback_value(cb_data, x[pointsMedians[i],pointsMedians[j]])
                end
            end
        end
    end

     # The global idea is to add the edges one after the other
     # in the order of the x_ij values sorted from the highest to the lowest
     # Adding an edge is valid only 
     # if the edge linked two nodes having a degree < 2 in the solution
     # and if the edge does not form a subtour  (i.e. a cycle of size < n nodes)
     # the detection of the creation of a cycle is done by the techniques     
     # called "union-find" structure where each node is associated with the number
     # of the smallest index of a node linked by a path
     # each time an edge is added, this number (call the connected component)
     # must be updated
        
     sol=zeros(Float64,G.nb_points, G.nb_points);
        
     L=[]
     for i in 1:nb_p
         for j in i+1:nb_p
           push!(L,(i,j,xfrac[i,j]))
         end
     end
     sort!(L,by = x -> x[3])  
       
     CC= zeros(Int64,nb_p);  #Connected component of node i
     for i in 1:nb_p
        CC[i]=-1
     end

     tour=zeros(Int64,nb_p,2)  # the two neighbours of i in a TSP tour, the first is always filled before de second
     for i in 1:nb_p
         tour[i,1]=-1
         tour[i,2]=-1
     end
     
     cpt=0
     while ( (cpt!=nb_p-1) && (size(L)!=0) )
     
        (i,j,val)=pop!(L)   

        if ( ( (CC[i]==-1) || (CC[j]==-1) || (CC[i]!=CC[j]) )  && (tour[i,2]==-1) && (tour[j,2]==-1) ) 
        
           cpt=cpt+1 
           
           if (tour[i,1]==-1)  # if no edge going out from i in the sol
           	tour[i,1]=j        # the first outgoing edge is j
	        CC[i]=i;
           else
         	tour[i,2]=j        # otherwise the second outgoing edge is j
           end

           if (tour[j,1]==-1)
        	tour[j,1]=i
         	CC[j]=CC[i]
           else
        	tour[j,2]=i
        	
        	oldi=i
 	        k=j
        	while (tour[k,2]!=-1)  # update to i the CC of all the nodes linked to j
        	  if (tour[k,2]==oldi) 
        	     l=tour[k,1]
              else 
                 l=tour[k,2]
              end
        	  CC[l]=CC[i]
 	          oldi=k
        	  k=l
        	end
	      end
        end
     end
     
     i1=-1          # two nodes haven't their 2nd neighbour encoded at the end of the previous loop
     i2=0
     for i in 1:nb_p
      if tour[i,2]==-1
        if i1==-1
           i1=i
        else 
           i2=i
        end
      end
     end
     tour[i1,2]=i2
     tour[i2,2]=i1
    
     value=0
     for i in 1:nb_p
       for j in i+1:nb_p     
         if ((j!=tour[i,1])&&(j!=tour[i,2]))
           sol[pointsMedians[i],pointsMedians[j]]=0
         else          
           sol[pointsMedians[i],pointsMedians[j]]=1      
      	   value=value+dist(G,pointsMedians[i],pointsMedians[j])
         end
       end
     end
      
     xvec=vcat([m[:x][i, j] for i = 1:G.nb_points for j = i+1:G.nb_points])
     solvec=vcat([sol[i, j] for i = 1:G.nb_points for j = i+1:G.nb_points])

     MathOptInterface.submit(m, MathOptInterface.HeuristicSolution(cb_data), xvec, solvec)
    
   end
  #
  #################

  #################
  # Setting callback in CPLEX
    # our lazySep_ViolatedAcyclic function sets a LazyConstraintCallback of CPLEX
    MathOptInterface.set(m, MathOptInterface.LazyConstraintCallback(), lazySep_ViolatedMengerCut) 
    
    # our userSep_ViolatedAcyclic function sets a LazyConstraintCallback of CPLEX   
    MathOptInterface.set(m, MathOptInterface.UserCutCallback(), userSep_ViolatedMengerCut)
    
    # our primal heuristic to "round up" a primal fractional solution
    MathOptInterface.set(m, MathOptInterface.HeuristicCallback(), primalHeuristicTSP)
  #
  #################


   println("Résolution du PLNE par le solveur")
   optimize!(m)
   println("Fin de la résolution du PLNE par le solveur")
   	
   #println(solution_summary(m, verbose=true))

   status = termination_status(m)

   # un petit affichage sympathique
   if status == MathOptInterface.OPTIMAL
      println("Valeur optimale = ", objective_value(m))
      println("Solution primale optimale :")
      
      #for i= 1:G.nb_points
      #   for j= i+1:G.nb_points
      #      println("x(",i,",",j,")=",value(x[i,j]))
      #   end
      #end
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

      S= find_cycle_in_integer_x(x, 1)
      push!(S,first(S))
      
      println("Temps de résolution :", solve_time(m))
      println("Number of generated Menger Cut constraints  : ", nbViolatedMengerCut_fromIntegerSep+nbViolatedMengerCut_fromFractionalSep)
      println("   from IntegerSep : ", nbViolatedMengerCut_fromIntegerSep)
      println("   from FractionalSep :", nbViolatedMengerCut_fromFractionalSep)

      return stations, affectations,S
    else
      println("Problème lors de la résolution")
    end
     
  
end





# function ring_star_ncompact(G, p)

#     c = calcul_dist(G)
    
# 	println("Création du PLNE")
#     m = Model(CPLEX.Optimizer)

#     @variable(m, y[1:G.nb_points, 1:G.nb_points], Bin)

#     @variable(m, x[1:G.nb_points, 1:G.nb_points] , Bin)

#     @objective(m, Min, sum(sum(c[i, j] * x[i, j] for j in 1:G.nb_points) for i in 1:G.nb_points) + sum(sum(c[i, j] * y[i, j] for j in 1:G.nb_points) for i in 1:G.nb_points))

#     #pas d'arete i à i
#     for i in 1:G.nb_points
#         @constraint(m, x[i, i] == 0)
#     end

#     #contraintes de stations
#     @constraint(m, y[1, 1] == 1)

#     for j in 2:G.nb_points
# 		@constraint(m, y[1, j] == 0)
# 	end

#     @constraint(m, sum(y[i, i] for i in 1:G.nb_points) == p) #1

#     for i in 1:G.nb_points
#         @constraint(m, sum(y[i, j] for j in 1:G.nb_points) == 1) #2
#     end

#     for i in 1:G.nb_points
#         for j in 1:G.nb_points
#             if i!=j
#                 @constraint(m, y[i, j] <= y[j, j]) #(3)
#             end
#         end
#     end

#     for i in 1:G.nb_points
#         @constraint(m, sum(x[j, i] for j in 1:i-1) + sum(x[i, j] for j in i+1:G.nb_points) == 2*y[i, i]) #4
#     end

#     for S in sous_ensembles([i for i in 2:G.nb_points], 2)
#         V_prive_S = setdiff([i for i in 1:G.nb_points], S)
#         deltaS = couples_possibles(S, V_prive_S)
#         @constraint(m, sum(sum(x[i, j] for (i,j) in e) for e in sous_ensembles(deltaS, 1)) >= 2 * sum(sum(y[i, i] for j in S) for i in S))
#     end

#     for i in 2:G.nb_points
#         for j in 1:G.nb_points
#             @constraint(m, y[j, j] >= x[i, j]) #9
#         end
#     end

#     # for W in sous_ensembles([i for i in 2:G.nb_points], 2)
#     #     V_prive_W = setdiff([i for i in 1:G.nb_points], W)
#     #     deltaW = couples_possibles(W, V_prive_W)
#     #     @constraint(m, sum(sum(x[i, j] for (i,j) in e) for e in sous_ensembles(deltaW, 1)) >= 2 * sum(sum(y[i, j] for j in W) for i in W))
#     # end

#     print(m)
# 	println()
	
# 	println("Résolution du PLNE par le solveur")
# 	optimize!(m)
#    	println("Fin de la résolution du PLNE par le solveur")
   	
# 	#println(solution_summary(m, verbose=true))

# 	status = termination_status(m)
 
# 	# un petit affichage sympathique
# 	if status == MathOptInterface.OPTIMAL
# 		println("Valeur optimale = ", objective_value(m))
# 		println("Solution primale optimale :")
		
#         stations = Int64[]
#         for i in 1:G.nb_points
#             if (value(y[i, i]) > 0.999)
#                 push!(stations, i)
#             end
#         end

#         println("Stations = ", stations)

#         affectations = []
#         for station in stations
#             l = Int64[]
#             push!(l, station)
#             for i in 1:G.nb_points
#                 if i!=station
#                     if (value(y[i, station]) > 0.999)
#                         push!(l, i)
#                     end
#                 end
#             end
#             push!(affectations, l)
#         end

#         println("Affectations =", affectations)

# 		cycle = Int64[]
#         i=1
#         j=2
#         while (value(x[i, j]) < 0.999) 
#      	    j=j+1
#         end        
#         push!(cycle,1)
#         push!(cycle,j)
#         i=j
#         tmp=1
#         while (i!=1)
#             j=1
#             while  ( j==i || (value(x[i,j]) < 0.999 && value(x[j,i]) < 0.999) || j==tmp ) 
#                 j=j+1
#             end
#             push!(cycle,j)
#             tmp=i
#             i=j
# 		end
#         println("S = ", cycle)
#         println("Temps de résolution :", solve_time(m))

#         return stations, affectations, cycle
# 	else
# 		 println("Problème lors de la résolution")
# 	end

# end

# function sous_ensembles(S, n)
#     #retourne les sous ensembles de S de taille >= n
#     sous_ensembles = collect(powerset(S))
#     return filter(x -> length(x) >= n, sous_ensembles)
# end

# function couples_possibles(l1, l2)
#     resultats = []
#     for e1 in l1
#         for e2 in l2
#             push!(resultats, (e1, e2))
#         end
#     end
#     return resultats
# end