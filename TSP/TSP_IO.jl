# Code largement inspiré du travail de Florian Belhassen-Dubois
using Plots

# Sructure de données "Nuage de points"
struct NuagePoints
	nb_points    # nombre de points
	X            # tableau des coordonnées x des points
	Y            # tableau des coordonnées y des points    
	max_distance # plus longue distance entre deux points
end

# Lecture d'un fichier de la TSPLIB .tsp (atttention uniquement les instances symétriques données par des coordonnées)
# Renvoie l'instance du TSP c
function Read_undirected_TSP(filename)
	
	I = NuagePoints(0, [], [],0)

	open(filename) do f
			
		node_coord_section = 0 # repère dans quelle section du fichier .tsp on est en train de lire
		nbp = 0
		X = Array{Float64}(undef, 0)
		Y = Array{Float64}(undef, 0)
			
		# on lit une par une nos ligne du fichier	
		for (i,line) in enumerate(eachline(f))
			
			# on sépare cette ligne en mots
			x = split(line," ") # For each line of the file, splitted using space as separator
             
			# on supprime les mots vides, en effet une espace suivi d'un autre espace renvoie le mot vide
			deleteat!(x, findall(e->e=="", x))

			
			if(node_coord_section == 0)       # If it's not a node section
					
				# si on est dans les coordonnées on va s'arrêter et remplir notre instance sinon il nous reste des labels à lire
				if(x[1] == "NODE_COORD_SECTION")
					node_coord_section = 1
				# si on est dans le label dimension, on le récupère
				elseif(x[1] == "DIMENSION")
					nbp = parse(Int, x[3])
				end
			
			# on est enfin dans nos coordonnées ! On les lit et on remplit notre instance avec
			elseif(node_coord_section == 1 && x[1] != "EOF")
				
				push!(X, parse(Float64, x[2]))
				push!(Y, parse(Float64, x[3]))
				
			else
				
				node_coord_section = 2
				
			end
		end
		
		
		# Calcule la plus longue distance entre deux points
		max_distance=0
	       for i in 1:nbp
	          for j in 1:nbp
	             if (max_distance < ( (X[i] - X[j])^2 + (Y[i] - Y[j])^2 ) )
	                max_distance = (X[i] - X[j])^2 + (Y[i] - Y[j])^2
	             end
	          end
	       end
	       
      		# on construit notre nuage de points
		I = NuagePoints(nbp, X, Y, max_distance)

		
	end
	return I
end

# Visualisation d'une instance comme un nuage de points dans un fichier pdf dont le nom est passé en paramètre
function WritePdf_visualization_TSP(I, filename)

	filename_splitted_in_two_parts = split(filename,".") # split to remove the file extension
	filename_with_pdf_as_extension= filename_splitted_in_two_parts[1]*".pdf"
	# save to pdf
	
	# un simple plot en fonction des coordonnées 
	p = plot(I.X, I.Y, seriestype = :scatter)
	savefig(p, filename_with_pdf_as_extension)

end


# Renvoie la distance euclidienne entre deux points du nuage
function dist(I, i, j) 
	
	return ((I.X[i] - I.X[j])^2 + (I.Y[i] - I.Y[j])^2)^(0.5)
	
end

# Crée une matrice de toutes les distances point à point
function calcul_dist(I)

	c = Array{Float64}(undef, (I.nb_points, I.nb_points))
	
	for i in 1:I.nb_points
	
		for j in 1:I.nb_points
		
			c[i, j] = dist(I, i, j)
		
		end
	
	end
	
	return c
	
end

 
# calcule la somme des coûts de notre arête solution
function Compute_value_TSP(I, S)
	
	res = ((I.X[S[1]] - I.X[S[I.nb_points]])^2 + (I.Y[S[1]] - I.Y[S[I.nb_points]])^2)^(0.5)
	for i = 1:(I.nb_points -1)	
		res = res + ((I.X[S[i]] - I.X[S[i+1]])^2 + (I.Y[S[i]] - I.Y[S[i+1]])^2)^(0.5)
	end
	
	return res

end

# calcule la somme des coûts de notre arête solution
function Compute_value_cycleRingStar(I, S)
	res = ((I.X[S[1]] - I.X[S[end]])^2 + (I.Y[S[1]] - I.Y[S[end]])^2)^(0.5)
	for i = 1:(length(S) -1)	
		res = res + ((I.X[S[i]] - I.X[S[i+1]])^2 + (I.Y[S[i]] - I.Y[S[i+1]])^2)^(0.5)
	end
	
	return res

end

# calcule le coût de notre solution
function Cout(I, S, affectations)
	res = 0
	for aff_i in affectations
		i = aff_i[1]
		for j in aff_i
			if (i != j)
				res = res + 10*dist(I, i, j) 
			end
		end
	end
	return res + Compute_value_cycleRingStar(I, S)
end

#pour un point p cherche il est affecte a quelle station
function affectation_de_p(affectations, p)
	for aff_i in affectations
		i = aff_i[1]
		if (i == p) || (p in aff_i)
			return i
		end
	end
	return 0
end

#calcule le cout de la distance min entre d et f dans le metro
function cout_dist_metro(I, S, d, f)
    len_cycle = length(S)
    ind_d = findfirst(x -> x == d, S)
    ind_f = findfirst(x -> x == f, S)

	direct = 0
	undirect = 0
	if (ind_d <= ind_f)
		for i in ind_d:ind_f-1
			direct = direct + dist(I, S[i], S[i+1])
		end
		for i in ind_f:len_cycle-1
			undirect = undirect + dist(I, S[i], S[i+1])
		end
		for i in 1:ind_d-1
			undirect = undirect + dist(I, S[i], S[i+1])
		end
	end
	if (ind_d > ind_f)
		for i in ind_f:ind_d-1
			direct = direct + dist(I, S[i], S[i+1])
		end
		for i in ind_d:len_cycle-1
			undirect = undirect + dist(I, S[i], S[i+1])
		end
		for i in 1:ind_f-1
			undirect = undirect + dist(I, S[i], S[i+1])
		end
	end
    return min(direct,undirect)
end

#calcule le cout de marche entre d et f
function cout_dist_marche(I, affectations, d, f)
	aff_d = affectation_de_p(affectations, d)
	aff_f = affectation_de_p(affectations, f)
	return 10*dist(I, aff_d, d) + 10*dist(I, aff_f, f)
end

#calcule la moyenne des couts des distances min entre tous les points d'une solution
function Mean_cout_dist_min(I, S, affectations)
	nb_chemins = 0
	res = 0
	for d in 1:I.nb_points
		for f in d+1:I.nb_points
			aff_d = affectation_de_p(affectations, d)
			aff_f = affectation_de_p(affectations, f)
			res = res + cout_dist_metro(I, S, aff_d, aff_f) + cout_dist_marche(I, affectations, d, f)
			nb_chemins = nb_chemins + 1
		end
	end
	return res / nb_chemins
end

function Ratio(I, S, affectations)
	nb_chemins = 0
	res = 0
	for d in 1:I.nb_points
		for f in d+1:I.nb_points
			aff_d = affectation_de_p(affectations, d)
			aff_f = affectation_de_p(affectations, f)
			cout_metro = cout_dist_metro(I, S, aff_d, aff_f)
			if (cout_metro == 0)
				continue
			else
				res = res + (cout_dist_marche(I, affectations, d, f) / cout_metro)
				nb_chemins = nb_chemins + 1
			end
		end
	end
	return res / nb_chemins
end
# permet de visualiser notre solution (un circuit / cycle) dans un fichier pdf dont le nom est spécifié en paramètres
# La solution est donnée par la liste ordonné des points à visiter commençant par 1
function WritePdf_visualization_solution_ordre(I, S, affectations, filename)

	filename_splitted_in_two_parts = split(filename,".") # split to remove the file extension
	filename_with_pdf_as_extension= filename_splitted_in_two_parts[1]*".pdf"
	# save to pdf
	
	tabX= Float16[]
	tabY= Float16[]
	
    for i in S
       push!(tabX, I.X[i])
       push!(tabY, I.Y[i])
    end
	
	# on ajoute le premier point pour le plot, c'est important sinon il manque l'arête entre 1 et n...
	push!(tabX, I.X[1])
	push!(tabY, I.Y[1])
	
	p = plot(I.X, I.Y, seriestype = :scatter,legend = false)
	plot!(p, tabX, tabY,legend = false)

	for i in 1:size(affectations)[1]
		k = affectations[i]
		x_i, y_i = I.X[affectations[i][1]], I.Y[affectations[i][1]]
		for j in 2:size(k)[1]
			x_j, y_j = I.X[affectations[i][j]], I.Y[affectations[i][j]]
			plot!(p,[x_i, x_j], [y_i, y_j], line=:dash, color=:blue, legend = false)
		end
	end

	savefig(p, filename_with_pdf_as_extension)

end

