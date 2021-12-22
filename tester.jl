using Agents, Random, Distributions, InteractiveDynamics, Clustering, OpenStreetMapXPlot, Plots, Distances, DataStructures, LinearAlgebra;
gr()

struct Colet
    id::Int64
    destinatie::Tuple{Int,Int,Float64}
    volum_l::Float16
    valoare_ron::Float16
end

@agent Transportor OSMAgent begin
    timp_ramas_pas_anterior_s::Float16
    viteza::Int
    capacitate_l::Float16
    capacitate_consumata_l::Float16
    prob_livrare_corecta::Float16
    timpi_transfer_pachet_s::Vector{Float16}
    coada_colete_livrat::Queue{Int64}
    coada_colete_ridicat::Vector{Int64}
    id_colet_in_procesare::Int
    cost::Float64
    venit::Float64
    id_depozit::Int
    id_colet_de_livrat::Int
    id_alt_transportor::Int
end

@agent Depozitar OSMAgent begin
    timp_ramas_pas_anterior_s::Float16
    capacitate_l::Float16
    capacitate_consumata_l::Float16
    prob_livrare_corecta::Float16
    timpi_transfer_pachet_s::Vector{Float16}
    coada_transportori::Queue{Int64}
    coada_colete::Vector{Int64}
    cost::Float64
    venit::Float64
end

function cel_mai_plin_depozitar(agent, model)
    depozitari = [i for i in keys(model.depozitari)]
    idmax = [length(intersect(Set(model[d].coada_colete), Set(agent.coada_colete_ridicat))) for d in depozitari]
    idmax = findmax(idmax)[2]
    return depozitari[idmax]
end

#agent_sz(agent) = typeof(agent) == Colet ? 0 : 5
agent_color(agent) = typeof(agent) == Depozitar ? :black : :green

function plotagents(model)
    ids = model.scheduler(model)
    ids = [i for i in ids if typeof(model[i]) != Colet]
    colors = [agent_color(model[i]) for i in ids]
    #sizes = [agent_sz(model[i]) for i in ids]
    markers = :circle
    pos = [OSM.map_coordinates(model[i], model) for i in ids]

    scatter!(
        pos;
        markercolor = colors,
        markersize = 5,
        markershapes = markers,
        label = "",
        markerstrokewidth = 0.5,
        markerstrokecolor = :black,
        markeralpha = 0.7,
    )
end

function plotpaths(model)
    ids = model.scheduler(model)
    ids = [i for i in ids if typeof(model[i]) != Colet]
    colors = [agent_color(model[i]) for i in ids]
    #sizes = [agent_sz(model[i]) for i in ids]
    markers = :circle
    pos = [OSM.map_coordinates(model[i], model) for i in ids]

    scatter!(
        pos;
        markercolor = colors,
        markersize = 5,
        markershapes = markers,
        label = "",
        markerstrokewidth = 0.5,
        markerstrokecolor = :black,
        markeralpha = 0.7,
    )
end

function optimize_kmedoids_closest!(model)
    sz = length(model.colete)
    destinatii = zeros(Float64, sz, 2)
    for (i, v) in enumerate(values(model.colete))
        destinatii[i, :] .= OSM.latlon(v.destinatie, model)
    end
    destinatii = transpose(destinatii)
    R = pairwise(Euclidean(), destinatii, dims=2)
    res = kmedoids(R, length(model.transportori))
    centroid_mapping = Dict{Int, Vector{Int}}(i=> Vector{Int}() for i in res.medoids)
    for i in 1:sz
        push!(centroid_mapping[res.medoids[res.assignments[i]]], i)
    end
    
    medoid_asociat = Dict()
    destinatie_asociata = Dict()
    for id in model.transportori
        distanta = reshape([i for i in OSM.latlon(model[id].pos, model)], 2, 1)
        indecsi_disponibili = (x->collect.(x)).(setdiff(keys(centroid_mapping), keys(medoid_asociat)))
        index_ales = sample(indecsi_disponibili)[1]
        idx = argmin(pairwise(Euclidean(), destinatii[:, centroid_mapping[index_ales]], distanta, dims=2))[1]
        medoid_asociat[index_ales] = id
        destinatie_asociata[id] = idx
    end

    for k in keys(centroid_mapping)
        next = destinatie_asociata[medoid_asociat[k]]
        vals = copy(centroid_mapping[k])
        closest = zeros(Int64, length(vals))
        for i in 1:(length(centroid_mapping[k]) - 1)
            idx = argmin(R[next, vals])
            closest[i] = vals[idx]
            vals = deleteat!(vals, idx)
        end
        closest[length(closest)] = vals[1]
        centroid_mapping[k] = copy(closest)
    end
    it = 1
    for id in model.transportori
        model[id].coada_colete_ridicat = [999 + v for v in centroid_mapping[res.medoids[it]]]
        it += 1
    end
end

function optimize_kmedoids!(model)
    sz = length(model.colete)
    destinatii = zeros(Float64, sz, 2)
    for (i, v) in enumerate(values(model.colete))
        destinatii[i, :] .= OSM.latlon(v.destinatie, model)
    end
    destinatii = transpose(destinatii)
    R = pairwise(Euclidean(), destinatii, dims=2)
    res = kmedoids(R, length(model.transportori))
    centroid_mapping = Dict{Int, Vector{Int}}(i=> Vector{Int}() for i in res.medoids)
    for i in 1:sz
        push!(centroid_mapping[res.medoids[res.assignments[i]]], i)
    end

    for k in keys(centroid_mapping)
        new_order = sortperm(R[k, centroid_mapping[k]])
        centroid_mapping[k] = centroid_mapping[k][new_order]
    end
    it = 1
    for id in model.transportori
        model[id].coada_colete_ridicat = [999 + v for v in centroid_mapping[res.medoids[it]]]
        it += 1
    end
end

function episode_end(model, step_number)
    (step_number > 86_400) && println(model.rng_seed, " ", step_number)
    (length(model.agents) == 0) || (step_number > 86_400)
end
function  model_step!(model)
    model.timp_trecut_s += model.dim_pas_s
end

function initialise(; map_path::String= "map.osm", n_depozitari::Int=2, n_transportori::Int=3, n_colete::Int=20, lungime_pas_s::Int=30, rng_seed=false, debug=false)
    properties = Dict()
    if typeof(rng_seed) != Bool
        properties[:rng_seed] = rng_seed
        Random.seed!(rng_seed)
    end
    properties[:colete] = Dict{Int64, Colet}()
    properties[:depozitari] = Dict{Int64, Tuple{Int,Int,Float64}}()
    properties[:transportori] = Vector{Int}()
    properties[:profit] = Dict{Int64, Float64}()
    properties[:timp_trecut_s] = 0
    properties[:dim_pas_s] = lungime_pas_s
    properties[:debug] = debug
    model = ABM(Union{Transportor, Depozitar}, OpenStreetMapSpace(map_path); properties=properties, warn=false)
    d_prob_livrare_corecta = truncated(Normal(.8, 2), .4, 1.)
    d_capacitate = truncated(Normal(20, 10), 5, 100)
    d_timpi_transfer = truncated(Exponential(60), 5, 60*15)
    d_viteza = truncated(Normal(10, 5), 2, 25)
    puncte_pornire = Vector{Tuple{Int,Int,Float64}}()
    # Deppzitari
    for id in 1:n_depozitari
        adresa = OSM.random_position(model)
        push!(puncte_pornire, adresa)
        ruta = OSM.plan_route(adresa, adresa, model)
        capacitate = rand(d_capacitate)
        prob_livrare = rand(d_prob_livrare_corecta)
        timp_transfer = rand(d_timpi_transfer, 1)
        depozitar = Depozitar(id, adresa, ruta, adresa, 0, capacitate, 0., prob_livrare,
            timp_transfer, Queue{Int64}(), [], 0, 0)
        add_agent_pos!(depozitar, model)
        model.depozitari[id] = adresa
    end
    # Colete
    d_volum = truncated(Normal(.1, .1), .01, 2)
    d_valoare = truncated(Normal(50, 50), 5, 10000)
    id_depozitari = collect(1:n_depozitari)
    for id in 1000:(1000 + n_colete - 1)
        sfarsit = OSM.random_road_position(model)
        volum = rand(d_volum)
        oid = sample(id_depozitari)
        model[oid].capacitate_consumata_l += volum
        valoare = rand(d_valoare)
        colet = Colet(id, sfarsit, volum, valoare)
        model.properties[:colete][id] = colet
        push!(model[oid].coada_colete, id)
    end
    
    # Transportori
    d_capacitate = truncated(Normal(5, 2), 1, 20)
    it = 1
    for id in (n_depozitari+1):(n_depozitari + n_transportori)
        inceput = OSM.random_road_position(model)
        ruta = []
        colete_de_ridicat = []
        colete_de_livrat = Queue{Int}()
        capacitate = rand(d_capacitate)
        prob_livrare = rand(d_prob_livrare_corecta)
        timp_transfer = rand(d_timpi_transfer, 1)
        viteza = trunc(Int, rand(d_viteza, 1)[1])
        transportor = Transportor(id, inceput, ruta, inceput, 0, viteza, capacitate, 0., 
            prob_livrare, timp_transfer, colete_de_livrat, colete_de_ridicat, -1, 0, 0, -1, 0, -1)
        add_agent_pos!(transportor, model)
        push!(model.transportori, id)
        it += 1
    end
    return model
end

function agent_step!(agent, model)
    # Step every 30 seconds
    step_time_s = model.dim_pas_s
    debug = model.debug
    timp_tr_mediu = mean(agent.timpi_transfer_pachet_s)
    d_timpi_transfer = truncated(Exponential(timp_tr_mediu), 5, 60*15)
    if (typeof(agent) == Depozitar)
        if length(agent.coada_colete) == 0
            model.profit[agent.id] = agent.venit - agent.cost
            delete!(model.depozitari, agent.id)
            kill_agent!(agent.id, model)
        end
        agent.timp_ramas_pas_anterior_s = (length(agent.coada_transportori) == 0) ? 0 : agent.timp_ramas_pas_anterior_s
        step_time_s -= agent.timp_ramas_pas_anterior_s
        while (length(agent.coada_transportori) > 0) & (step_time_s > 0)
            timp_incarcare = rand(d_timpi_transfer)[1]
            push!(agent.timpi_transfer_pachet_s, timp_incarcare)
            step_time_s = step_time_s - timp_incarcare
            agent.timp_ramas_pas_anterior_s = -step_time_s
            tr = dequeue!(agent.coada_transportori)
            colete = intersect(agent.coada_colete, model[tr].coada_colete_ridicat)
            debug && println("Depozitar $(agent.id) incarc transportorul $tr cu coletele $colete")
            agent.venit += sum([model.colete[i].valoare_ron*.005 for i in colete])
            agent.cost += sum([model.colete[i].volum_l*.002 for i in colete]) + 22/3600*step_time_s
            foreach(i -> enqueue!(model[tr].coada_colete_livrat, i), colete)
            model[tr].id_depozit = -1
            setdiff!(agent.coada_colete, colete)
            setdiff!(model[tr].coada_colete_ridicat, colete)
        end
    elseif typeof(agent) == Transportor
        agent.timp_ramas_pas_anterior_s = max(agent.timp_ramas_pas_anterior_s - step_time_s, 0)
        ans = is_stationary(agent, model) ? "da" : "nu"
        #debug && println("Transportor $(agent.id): Timp anterior ramas - $(agent.timp_ramas_pas_anterior_s), stau pe loc $ans")
        
        if agent.timp_ramas_pas_anterior_s <= 0 && is_stationary(agent, model)
            debug && println("Transportor $(agent.id): Timp anterior ramas - $(agent.timp_ramas_pas_anterior_s), stau pe loc $ans")
            # S-a ajuns la destinatie si se livreaza pachetul 
            if (agent.id_colet_de_livrat>0)
                timp_incarcare = rand(d_timpi_transfer)[1]
                push!(agent.timpi_transfer_pachet_s, timp_incarcare)
                agent.timp_ramas_pas_anterior_s = timp_incarcare
                agent.venit += model.colete[agent.id_colet_de_livrat].valoare_ron*.01
                delete!(model.colete, agent.id_colet_de_livrat)
                debug && println("Transportor $(agent.id): Am livrat $(agent.id_colet_de_livrat).")
                agent.id_colet_de_livrat = 0
            end
            # Am pachete la mine si trebuie sa-l livrez pe urmatorul
            if (length(agent.coada_colete_livrat) > 0) && (agent.id_colet_de_livrat==0)
                agent.id_colet_de_livrat = dequeue!(agent.coada_colete_livrat)
                debug && println("Transportor $(agent.id): Livrez $(agent.id_colet_de_livrat)")
                agent.destination = model.colete[agent.id_colet_de_livrat].destinatie
                agent.route = OSM.plan_route(agent.pos, agent.destination, model)
            end
            # Nu mai am colete si trebuie sa merg la un depozit sa iau
            if ((length(agent.coada_colete_livrat)==0) && (length(agent.coada_colete_ridicat) > 0)) && (agent.id_colet_de_livrat<=0)
                agent.id_depozit = cel_mai_plin_depozitar(agent, model)
                debug && println("Transportor $(agent.id): Iau pachete de la $(agent.id_depozit).")
                agent.destination = model.depozitari[agent.id_depozit]
                agent.route = OSM.plan_route(agent.pos, agent.destination, model)
                agent.id_colet_de_livrat = -1
            end
            # Sunt la depozit si trebuie sa stau la coada ca sa fiu incarcat cu colete
            if (length(setdiff([agent.id_depozit], nearby_ids(agent.pos, model, 10))) == 0) && (agent.id_colet_de_livrat == -1)
                enqueue!(model[agent.id_depozit].coada_transportori, agent.id)
                agent.id_colet_de_livrat = 0
            end
            # Nu mai am treaba si ma retrag
            if ((length(agent.coada_colete_ridicat) == 0) && (length(agent.coada_colete_livrat)==0)) && (agent.id_colet_de_livrat==0)
                model.profit[agent.id] = agent.venit - agent.cost
                delete!(model.colete, agent.id_colet_de_livrat)
                kill_agent!(agent.id, model)
            end
        end
        distanta_parcursa = step_time_s * agent.viteza
        agent.cost += distanta_parcursa * 7/100_000 * 6 + 22/3600*(step_time_s)
        move_along_route!(agent, model, distanta_parcursa)
    end
end

n_depozitari=3
n_transportori=6
n_colete=100
lungime_pas_s = 1

m1_means = Vector{Float16}()
m2_means = Vector{Float16}()
for i in 1:20
    m1 = initialise(;n_depozitari=n_depozitari, n_transportori=n_transportori, n_colete=n_colete,
        lungime_pas_s=lungime_pas_s, rng_seed=i)#, debug=true)
    optimize_kmedoids_closest!(m1)
    run!(m1, agent_step!, model_step!, episode_end)
    push!(m1_means, mean(values(m1.profit)))
    m2 = initialise(;n_depozitari=n_depozitari, n_transportori=n_transportori, n_colete=n_colete,
        lungime_pas_s=lungime_pas_s, rng_seed=i)
    optimize_kmedoids!(m2)
    run!(m2, agent_step!, model_step!, episode_end)
    push!(m2_means, mean(values(m2.profit)))
end
println(mean(m1_means))
println(mean(m2_means))