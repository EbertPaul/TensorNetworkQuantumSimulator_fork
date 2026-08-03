using Base.Threads
using ITensors: Algorithm
using Dictionaries: Dictionary, set!

function default_residual_tol() :: Float64 return 1e-7 end
function default_residual_diff_tol() :: Float64 return 1e-10 end
function default_msgdiff_tol() :: Float64 return 1e-15 end


mutable struct BeliefPropagationCacheHistory{V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    current_BeliefPropagationCache::BeliefPropagationCache{V, N, M}      
    vec_edge_sequence::AbstractVector{<:NamedEdge}                      # edge sequence that is use to vectorize message dictionaries (must not be the same as bpc edge sequence)
    message_proposal_history::Vector{Vector{M}}                         # ring buffer containing proposals
    message_residues_history::Vector{Vector{M}}                         # ring buffer containing residues
    history_capacity::Int                                               # number of slots in ring buffer
    history_len::Int                                                    # number of valid entries currently stored
    history_head::Int                                                   # index of most recent entry (0 means empty)
end

function BeliefPropagationCacheHistory(
        bpc::BeliefPropagationCache{V, N, M};
        history_capacity::Int = 10,
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    if history_capacity <= 0
        error("history_capacity must be a positive integer.")
    end
    edges = collect(edge_sequence(bpc))
    return BeliefPropagationCacheHistory(
        bpc,
        edges,
        Vector{Vector{M}}(undef, history_capacity),
        Vector{Vector{M}}(undef, history_capacity),
        history_capacity,
        0,
        0,
    )
end

function BeliefPropagationCache(bpch::BeliefPropagationCacheHistory)
    return bpch.current_BeliefPropagationCache
end

function vec_edge_sequence(bpch::BeliefPropagationCacheHistory)
    return bpch.vec_edge_sequence
end

@inline function ring_index(head::Int, iterations_into_past::Int, capacity::Int)
    return mod1(head - iterations_into_past + 1, capacity)
end

function history_length(bpch::BeliefPropagationCacheHistory)
    cl = bpch.history_len
    if cl < 0 || cl > bpch.history_capacity
        error("History length is inconsistent with history capacity. Got history_len=$(bpch.history_len), history_capacity=$(bpch.history_capacity).")
    end
    return cl
end

function past_message_proposals(bpch::BeliefPropagationCacheHistory, iterations_into_past::Int)
    if iterations_into_past <= 0
        error("When accessing prior message proposals iterations_into_past must be a positive integer, where 1 corresponds to the most recent proposal that lead to the current state of the BeliefPropagationCache.")
    end
    if iterations_into_past <= history_length(bpch)
        idx = ring_index(bpch.history_head, iterations_into_past, bpch.history_capacity)
        return bpch.message_proposal_history[idx]
    else
        error("The stored message proposal history contains $(history_length(bpch)) iterations, but you requested a message proposal from $(iterations_into_past) iterations into the past.")
    end
end

function past_message_residues(bpch::BeliefPropagationCacheHistory, iterations_into_past::Int)
    if iterations_into_past <= 0
        error("When accessing prior message residues iterations_into_past must be a positive integer, where 1 corresponds to the most recent residue that lead to the current state of the BeliefPropagationCache.")
    end
    if iterations_into_past <= history_length(bpch)
        idx = ring_index(bpch.history_head, iterations_into_past, bpch.history_capacity)
        return bpch.message_residues_history[idx]
    else
        error("The stored message residues history contains $(history_length(bpch)) iterations, but you requested a message residue from $(iterations_into_past) iterations into the past.")
    end
end

function push_history!(
        bpch::BeliefPropagationCacheHistory{V, N, M},
        new_message_proposals::Vector{M},
        new_message_residues::Vector{M},
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    cap = bpch.history_capacity
    if bpch.history_len < cap
        idx = bpch.history_len + 1
        bpch.history_len += 1
    else
        idx = (bpch.history_head % cap) + 1
    end
    bpch.message_proposal_history[idx] = new_message_proposals
    bpch.message_residues_history[idx] = new_message_residues
    bpch.history_head = idx
    return bpch
end

# update BeliefPropagationCacheHistory with a new BeliefPropagationCache (can have new network and new messages)
# and the new message proposals and residuals associated with the new set of messages
function update_history(
        bpch::BeliefPropagationCacheHistory,
        new_bpc::BeliefPropagationCache,
        new_message_proposals::Vector{M},
        new_message_residues::Vector{M};
    ) where M <: Union{ITensor, Vector{ITensor}}

    new_edge_sequence = collect(edge_sequence(new_bpc))
    if new_edge_sequence != vec_edge_sequence(bpch)
        error("Cannot update BeliefPropagationCacheHistory with new BeliefPropagationCache: edge_sequence changed.")
    end
    bpch.current_BeliefPropagationCache = new_bpc
    return push_history!(bpch, new_message_proposals, new_message_residues)
end

# update message/residual/proposal history of BeliefPropagationCacheHistory, while underlying network is untouched
function update_history(
        bpch::BeliefPropagationCacheHistory,
        new_accepted_messages::Vector{M},
        new_message_proposals::Vector{M},
        new_message_residues::Vector{M};
    ) where M <: Union{ITensor, Vector{ITensor}}
    new_bpc = update_bpc_messages(bpch.current_BeliefPropagationCache, new_accepted_messages, vec_edge_sequence(bpch))
    return update_history(
        bpch,
        new_bpc,
        new_message_proposals,
        new_message_residues;
    )
end

# update messages/residual/proposal history in the case where new_message_proposal = new_accepted_message
# e.g. when greedy update is used and its history shall be recorded
function update_history(
        bpch::BeliefPropagationCacheHistory,
        new_message_proposals::Vector{M},
        new_message_residues::Vector{M};
    ) where M <: Union{ITensor, Vector{ITensor}}
    return update_history(
        bpch,
        new_message_proposals,
        new_message_proposals,
        new_message_residues;
    )
end


#=
function update_bpc_messages(
        bpc::BeliefPropagationCache{V, N, M},
        new_messages::Vector{M}
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}

    es = edge_sequence(bpc)
    if length(new_messages) != length(es)
        error("Encountered message vector of length $(length(new_messages)) but edge_sequence has length $(length(es)).")
    end
    for i in eachindex(es)
        setmessage!(bpc, es[i], new_messages[i])
    end
    return bpc
end
=#

function update_bpc_messages(
        bpc::BeliefPropagationCache{V, N, M},
        new_messages::Vector{M},
        vec_edge_seq::AbstractVector{<:NamedEdge}
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    if length(new_messages) != length(vec_edge_seq)
        error("Encountered message vector of length $(length(new_messages)) but edge_sequence has length $(length(vec_edge_seq)).")
    end
    new_messages_dict = Dictionary{NamedEdge, M}()
    for i in eachindex(vec_edge_seq)
        set!(new_messages_dict, vec_edge_seq[i], new_messages[i])
    end
    return BeliefPropagationCache(
        network(bpc),
        new_messages_dict,
        contraction_sequences(bpc),
        vec_edge_seq
    )
end

function identicalize_msg_inds(
    m::ITensor,
    temp_inds::Vector
)
    msg_inds = collect(ITensors.inds(m))
    if length(temp_inds) != 2 || length(msg_inds) != 2
        @warn "Template indices = $(temp_inds), message indices = $(msg_inds)."
        error("Cannot align message indices with template: both must have exactly 2 indices.")
    end

    n_temp_primed = count(i -> plev(i) > 0, temp_inds)
    n_msg_primed = count(i -> plev(i) > 0, msg_inds)
    if n_temp_primed != 1
        error("Template must contain exactly one primed and one unprimed index.")
    end
    if n_msg_primed != 1
        error("Message must contain exactly one primed and one unprimed index.")
    end
    # all consistent, now do we have to transpose or not?
    template_prime_first = plev(temp_inds[1]) > 0
    msg_prime_first = plev(msg_inds[1]) > 0
    if template_prime_first == msg_prime_first
        out = ITensors.replaceinds(m, msg_inds, temp_inds)
    else
        out = ITensors.replaceinds(m, reverse(msg_inds), temp_inds)
    end
    return ITensors.permute(out, temp_inds...)
end

# subtract messages m_a - m_b and return with indices of m_a
function index_safe_message_subtract(m_a::ITensor, m_b::ITensor)
    m_b_aligned = identicalize_msg_inds(m_b, collect(ITensors.inds(m_a)))
    return m_a - m_b_aligned
end


function predictor_step_initialization(
        current_msgs::AbstractVector{M1},
        past_msgs::AbstractVector{M2},
        this_step::Float64,
        last_step::Float64,
        vec_edge_seq::AbstractVector{<:NamedEdge},
    ) where {M1 <: Union{ITensor, Vector{ITensor}}, M2 <: Union{ITensor, Vector{ITensor}}}
    n_edges = length(current_msgs)
    if length(past_msgs) != n_edges
        error("Encountered message vectors of different lengths: $(length(current_msgs)) and $(length(past_msgs)).")
    end

    predictor_msgs = Vector{M1}(undef, n_edges)
    α = this_step / last_step
    #Threads.@threads :greedy for e_idx in 1:n_edges
    for e_idx in 1:n_edges
        predictor_msgs[e_idx] = current_msgs[e_idx] + α * index_safe_message_subtract(current_msgs[e_idx], past_msgs[e_idx])
        predictor_msgs[e_idx] = make_hermitian(predictor_msgs[e_idx])
        predictor_msgs[e_idx] /= tr(predictor_msgs[e_idx])
    end
    return predictor_msgs
end

function predictor_step_initialization(
        current_bpc::BeliefPropagationCache{V, N, M},
        past_msgs::Union{Nothing, AbstractVector{<:Union{ITensor, Vector{ITensor}}}},
        this_step::Float64,
        last_step::Float64,
        vec_edge_seq::AbstractVector{<:NamedEdge};
        history_capacity::Int = 10,
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    # without past messages, not predictor step can be done, revert to copying messages from last temperature
    if isnothing(past_msgs)
        return BeliefPropagationCacheHistory(current_bpc, history_capacity = history_capacity)
    end
    current_msgs = [message(current_bpc, e) for e in vec_edge_seq] 
    # perform the step
    predictor_msgs = predictor_step_initialization(
        current_msgs,
        past_msgs,
        this_step,
        last_step,
        vec_edge_seq,
    )
    predictor_msgs_typed = Vector{M}(undef, length(vec_edge_seq))
    for i in eachindex(vec_edge_seq)
        predictor_msgs_typed[i] = convert(M, predictor_msgs[i])
    end

    new_bpc = update_bpc_messages(current_bpc, predictor_msgs_typed, vec_edge_seq)
    return BeliefPropagationCacheHistory(new_bpc, history_capacity = history_capacity)
end

function simultaneous_greedy_update(
    bpc::BeliefPropagationCache{V, N, M},
    es::AbstractVector{<:NamedEdge};
    update_alg = set_default_kwargs(Algorithm(default_message_update_alg(bpc)), bpc)
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    n = length(es)
    message_proposal_vec = Vector{M}(undef, n) # use vector for thread safety
    message_residue_vec = Vector{M}(undef, n) # use vector for thread safety
    # carry out contractions in prallel threads
    Threads.@threads :greedy for i in eachindex(es)
        e = es[i]
        new_message, _ = updated_message(update_alg, bpc, e)
        old_message = message(bpc, e)
        message_proposal_vec[i] = new_message
        message_residue_vec[i] = index_safe_message_subtract(new_message, old_message)
    end
    return message_proposal_vec, message_residue_vec
end

function anderson_least_squares(
        residues::Vector{Vector{M}},
        edges::AbstractVector{<:NamedEdge};
        threaded = true
    ) where M <: Union{ITensor, Vector{ITensor}}
    N = length(residues) # M + 1 = number of message proposals from the past + the current one
    C = zeros(N + 1, N + 1)
    if !threaded
        for i in 1:N
            for j in i:N
                Cij = 0.0
                for e_idx in eachindex(edges)
                    Cij += real(scalar(dag(residues[i][e_idx]) * residues[j][e_idx]))
                end
                C[i, j] = Cij
                C[j, i] = Cij
            end
        end
    else
        
        Threads.@threads :greedy for i in 1:N
            for j in i:N
                Cij = 0.0
                for e_idx in eachindex(edges)
                    Cij += real(scalar(dag(residues[i][e_idx]) * residues[j][e_idx]))
                end
                C[i, j] = Cij
                C[j, i] = Cij
            end
        end
    end
    C[N + 1, 1:N] .= 1.0 # extend C to incorporate the sum(alpha) = 1 contraint
    C[1:N, N + 1] .= 1.0
    C[N + 1, N + 1] = 0.0
    rhs = zeros(N + 1)
    rhs[N + 1] = 1.0
    sol = C \ rhs # sol = (alpha, mu) for one Lagrange multiplier mu
    alpha = sol[1:N]
    return alpha
end

function update_step_with_anderson_acceleration(
        bpch::BeliefPropagationCacheHistory{V, N, M},
        memory_window::Int, # how many of the previous message updates should be used to construct the Anderson-accelerated update
        ;
        alphas = nothing,
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}

    vec_edge_seq = vec_edge_sequence(bpch)
    n_edges = length(vec_edge_seq)
    bpc = bpch.current_BeliefPropagationCache
    update_alg = set_default_kwargs(Algorithm(default_message_update_alg(bpc)), bpc)
    new_message_proposals, new_message_residues = simultaneous_greedy_update(bpc, vec_edge_seq; update_alg = update_alg)
    n_prev = min(memory_window, history_length(bpch)) # number of previous update steps that are taken into account
    
    proposals = Vector{Vector{M}}(undef, n_prev + 1)
    residues = Vector{Vector{M}}(undef, n_prev + 1)
    proposals[1] = new_message_proposals
    residues[1] = new_message_residues
    for i in 1:n_prev
        proposals[i + 1] = past_message_proposals(bpch, i)
        residues[i + 1] = past_message_residues(bpch, i)
    end

    # solve the least-squares problem to find optimal linear combination of the current and the M previous message proposals
    alpha = anderson_least_squares(residues, vec_edge_seq)
    if !isnothing(alphas)
        push!(alphas, copy(alpha))
    end

    # construct the Anderson-accelerated message update and accept directly, although it may not be PD
    new_accepted_messages = Vector{M}(undef, n_edges)
    Threads.@threads :greedy for e_idx in 1:n_edges
        m_AA = sum(alpha[i] * proposals[i][e_idx] for i in 1:(n_prev + 1))
        m_AA = make_hermitian(m_AA)
        m_norm = tr(m_AA)
        new_accepted_messages[e_idx] = m_AA / m_norm
    end

    return new_accepted_messages, new_message_proposals, new_message_residues
end

function residuals_approx_zero(res::Vector{M}; tol::Float64 = 1e-8) where M <: Union{ITensor, Vector{ITensor}}
    for i in eachindex(res)
        if norm(res[i]) > tol
            return false
        end
    end
    return true
end

function residual_diff_approx_zero(
        res_a::Vector{M1},
        res_b::Vector{M2};
        tol::Float64 = 1e-8
    ) where {M1 <: Union{ITensor, Vector{ITensor}}, M2 <: Union{ITensor, Vector{ITensor}}}
    if length(res_a) != length(res_b)
        error("Encountered message vectors of different lengths: $(length(res_a)) and $(length(res_b)).")
    end
    for i in eachindex(res_a, res_b)
        if norm(res_a[i] - res_b[i]) > tol
            return false
        end
    end
    return true
end

function msgdiff_approx_zero(
        msgs_a::Vector{M1},
        msgs_b::Vector{M2};
        tol::Float64 = 1e-8
    ) where {M1 <: Union{ITensor, Vector{ITensor}}, M2 <: Union{ITensor, Vector{ITensor}}}
    if length(msgs_a) != length(msgs_b)
        error("Encountered message vectors of different lengths: $(length(msgs_a)) and $(length(msgs_b)).")
    end
    for i in eachindex(msgs_a)
        if message_diff(msgs_a[i], msgs_b[i]) > tol
            return false
        end
    end
    return true
end

function average_residual_norm(res::Vector{M}) where M <: Union{ITensor, Vector{ITensor}}
    total_norm = 0.0
    for i in eachindex(res)
        total_norm += norm(res[i])
    end
    return total_norm / length(res)
end

function maximum_residual_norm(res::Vector{M}) where M <: Union{ITensor, Vector{ITensor}}
    max_norm = 0.0
    for i in eachindex(res)
        max_norm = max(max_norm, norm(res[i]))
    end
    return max_norm
end

function average_message_diff(
        msgs_a::Vector{M1},
        msgs_b::Vector{M2}
    ) where {M1 <: Union{ITensor, Vector{ITensor}}, M2 <: Union{ITensor, Vector{ITensor}}}
    if length(msgs_a) != length(msgs_b)
        error("Encountered message vectors of different lengths: $(length(msgs_a)) and $(length(msgs_b)).")
    end
    total_diff = 0.0
    for i in eachindex(msgs_a)
        total_diff += message_diff(msgs_a[i], msgs_b[i])
    end
    return total_diff / length(msgs_a)
end

function messages_are_PD(messages::Vector{M}; threaded = false) where M <: Union{ITensor, Vector{ITensor}}
    if threaded
        Threads.@threads :greedy for i in eachindex(messages)
            m = messages[i]
            inds = collect(ITensors.inds(m))
            m_arr =  Array(m, inds...)
            if !isposdef(m_arr)
                return false
            end
        end
    else
        for i in eachindex(messages)
            m = messages[i]
            inds = collect(ITensors.inds(m))
            m_arr =  Array(m, inds...)
            if !isposdef(m_arr)
                return false
            end
        end
    end
    return true
end

function update_with_anderson_acceleration_cold_start(
        network::AbstractTensorNetwork;
        memory_window = 5,
        maxiter = 20,
        alphas = nothing,
        avg_residual_norms = nothing,
        max_residual_norms = nothing,
        avg_msg_diffs = nothing,
        check_PD = false,
        residual_tol::Float64 = default_residual_tol(),
        residual_diff_tol::Float64 = default_residual_diff_tol(),
        msgdiff_tol::Float64 = default_msgdiff_tol(),
    )

    # initialize the BeliefPropagationCache and the lower bound on the eigenvalues for each message
    bpc = BeliefPropagationCache(network) 
    edges = edge_sequence(bpc)
    for e in edges
        m_e = message(bpc, e)
        set!(bpc.messages, e, m_e) # enforces message initialization
    end

    # initialize the BeliefPropagationCacheHistory
    bpch = BeliefPropagationCacheHistory(bpc; history_capacity = memory_window)
    return update_with_anderson_acceleration(
        bpch;
        memory_window = memory_window,
        maxiter = maxiter,
        alphas = alphas,
        avg_residual_norms = avg_residual_norms,
        max_residual_norms = max_residual_norms,
        avg_msg_diffs = avg_msg_diffs,
        check_PD = check_PD,
        residual_tol = residual_tol,
        residual_diff_tol = residual_diff_tol,
        msgdiff_tol = msgdiff_tol,
    )
end


function update_with_anderson_acceleration(
        bpch::BeliefPropagationCacheHistory;
        memory_window = 10,
        maxiter = 100,
        alphas = nothing,
        avg_residual_norms = nothing,
        max_residual_norms = nothing,
        avg_msg_diffs = nothing,
        check_PD = false,
        initial_greedy_steps::Int = 0,
        residual_tol::Float64 = default_residual_tol(),
        residual_diff_tol::Float64 = default_residual_diff_tol(),
        msgdiff_tol::Float64 = default_msgdiff_tol(),
    )
    edges = vec_edge_sequence(bpch)
    # start performing Anderson-accelerated message updates and check for convergence or stagnation
    initial_messages = [message(bpch.current_BeliefPropagationCache, e) for e in edges]
    for i in 1:maxiter
        prev_messages = [message(bpch.current_BeliefPropagationCache, e) for e in edges]
        prev_residuals = i > 1 ? past_message_residues(bpch, 1) : nothing
        
        if i > initial_greedy_steps
            # perform anderson-accelerated update based on bpch
            new_accepted_messages, new_message_proposals, new_message_residues = update_step_with_anderson_acceleration(
                bpch,
                memory_window;
                alphas,
            )
            bpch = update_history(bpch, new_accepted_messages, new_message_proposals, new_message_residues)
        else
            # perform standard greedy update but record residues for the later history
            new_message_proposals, new_message_residues = simultaneous_greedy_update(bpch.current_BeliefPropagationCache, edges)
            bpch = update_history(bpch, new_message_proposals, new_message_residues)
            new_accepted_messages = new_message_proposals
        end

        # monitoring
        avg_residual_norm = average_residual_norm(new_message_residues)
        max_residual_norm = maximum_residual_norm(new_message_residues)
        avg_msg_diff = average_message_diff(new_accepted_messages, prev_messages)
        if !isnothing(avg_residual_norms)
            push!(avg_residual_norms, avg_residual_norm)
        end
        if !isnothing(max_residual_norms)
            push!(max_residual_norms, max_residual_norm)
        end
        if !isnothing(avg_msg_diffs)
            push!(avg_msg_diffs, avg_msg_diff)
        end
        
        # residuals = 0 implies BP fixed point -> test convergence
        if residuals_approx_zero(new_message_residues; tol=residual_tol)
            println("Converged after $i iterations (residuals approximately zero).")
            return bpch, true, new_accepted_messages, initial_messages
        end
        
        # check for stagnation based on messages (e.g. because the step size becomes too small)
        if msgdiff_approx_zero(new_accepted_messages, prev_messages; tol=msgdiff_tol)
            println("Stagnated after $i iterations (message approximately unchanged).")
            return bpch, false, new_accepted_messages, initial_messages
        end
        
        # check for stagnation based on residuals (e.g. because the step size becomes too small)
        if !isnothing(prev_residuals) && residual_diff_approx_zero(new_message_residues, prev_residuals; tol=residual_diff_tol)
            println("Stagnated after $i iterations (residual difference approximately zero).")
            return bpch, false, new_accepted_messages, initial_messages
        end

        check_PD_with_threading = false
        if check_PD && !messages_are_PD(new_accepted_messages; threaded=check_PD_with_threading)
            @warn "Messages no longer PD after iteration $i."
            return bpch, false, new_accepted_messages, initial_messages
        end
    end
end
