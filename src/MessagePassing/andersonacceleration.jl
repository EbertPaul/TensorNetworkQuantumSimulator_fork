using Base.Threads
using ITensors: Algorithm
using Dictionaries: Dictionary, set!

function default_residual_tol() :: Float64 return 1e-7 end
function default_residual_diff_tol() :: Float64 return 1e-10 end
function default_msgdiff_tol() :: Float64 return 1e-15 end

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

mutable struct BeliefPropagationCacheHistory{V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    bpc::BeliefPropagationCache{V, N, M}      
    vec_edge_sequence::AbstractVector{<:NamedEdge}                      # edge sequence that is use to vectorize message dictionaries (must not be the same as bpc edge sequence)
    message_history::Vector{Vector{M}}                                  # ring buffer containing history of previous messages
    residue_history::Vector{Vector{M}}                                  # ring buffer containing residues
    greedy_step_history::Vector{Vector{M}}                              # ring buffer containing history of messages obtained by applying a greedy update to the message from the message history
    history_capacity::Int                                               # number of slots in ring buffer
    history_len::Int                                                    # number of valid entries currently stored
    history_head::Int                                                   # index of most recent entry (0 means empty)
end


function BeliefPropagationCacheHistory(
        bpc::BeliefPropagationCache{V, N, M},
        x0,
        x1,
        u0,
        u1,
        r0,
        r1,
        vec_edge_seq::AbstractVector{<:NamedEdge} = collect(edge_sequence(bpc));
        history_capacity::Int = 10,
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    
    if history_capacity < 2
        error("history_capacity must be a positive integer greater than or equal to 2.")
    end
    message_history = Vector{Vector{M}}(undef, history_capacity)
    residue_history = Vector{Vector{M}}(undef, history_capacity)
    greedy_step_history = Vector{Vector{M}}(undef, history_capacity)
    message_history[1] = x0
    message_history[2] = x1
    residue_history[1] = r0
    residue_history[2] = r1
    greedy_step_history[1] = u0
    greedy_step_history[2] = u1
    return BeliefPropagationCacheHistory(
        bpc,
        vec_edge_seq,
        message_history,
        residue_history,
        greedy_step_history,
        history_capacity,
        2,
        2,
    )
end

function BeliefPropagationCache(bpch::BeliefPropagationCacheHistory)
    return bpch.bpc
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

function past_messages(bpch::BeliefPropagationCacheHistory, iterations_into_past::Int)
    if iterations_into_past <= 0
        error("When accessing prior messages iterations_into_past must be a positive integer, where 1 corresponds to the most recent message that lead to the current state of the BeliefPropagationCache.")
    end
    if iterations_into_past <= history_length(bpch)
        idx = ring_index(bpch.history_head, iterations_into_past, bpch.history_capacity)
        return bpch.message_history[idx]
    else
        error("The stored message proposal history contains $(history_length(bpch)) iterations, but you requested a message proposal from $(iterations_into_past) iterations into the past.")
    end
end

function past_residues(bpch::BeliefPropagationCacheHistory, iterations_into_past::Int)
    if iterations_into_past <= 0
        error("When accessing prior message residues iterations_into_past must be a positive integer, where 1 corresponds to the most recent residue that lead to the current state of the BeliefPropagationCache.")
    end
    if iterations_into_past <= history_length(bpch)
        idx = ring_index(bpch.history_head, iterations_into_past, bpch.history_capacity)
        return bpch.residue_history[idx]
    else
        error("The stored message residues history contains $(history_length(bpch)) iterations, but you requested a message residue from $(iterations_into_past) iterations into the past.")
    end
end

function past_greedy_steps(bpch::BeliefPropagationCacheHistory, iterations_into_past::Int)
    if iterations_into_past <= 0
        error("When accessing prior greedy message updates iterations_into_past must be a positive integer, where 1 corresponds to the most recent greedy update that lead to the current state of the BeliefPropagationCache.")
    end
    if iterations_into_past <= history_length(bpch)
        idx = ring_index(bpch.history_head, iterations_into_past, bpch.history_capacity)
        return bpch.greedy_step_history[idx]
    else
        error("The stored greedy message update history contains $(history_length(bpch)) iterations, but you requested a greedy message update from $(iterations_into_past) iterations into the past.")
    end
end

# assumes that new_messages are sorted according to vec_edge_sequence(bpch)!
function update_history!(
        bpch::BeliefPropagationCacheHistory{V, N, M},
        new_messages::Vector{M};
        return_msgdiff = false
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    # save prev_messages for later msgdiff computation if requested
    vec_edge_seq = vec_edge_sequence(bpch)
    prev_messages = return_msgdiff ? [message(BeliefPropagationCache(bpch), e) for e in vec_edge_seq] : nothing
    # handle indices for ring arrays
    cap = bpch.history_capacity
    if bpch.history_len < cap
        idx = bpch.history_len + 1
    else
        idx = (bpch.history_head % cap) + 1
    end
    # update bpc and get residues of the new messages
    new_bpc = update_bpc_messages(BeliefPropagationCache(bpch), new_messages, vec_edge_seq)
    update_alg = set_default_kwargs(Algorithm(default_message_update_alg(new_bpc)), new_bpc)
    greedy_steps, greedy_residues = simultaneous_greedy_update(new_bpc, vec_edge_seq; update_alg = update_alg)
    # compute monitoring diagnostics
    res_norms = norm.(greedy_residues)
    if return_msgdiff
        subtr_msg_diffs = Vector{Float64}(undef, length(vec_edge_seq))
        dot_msg_diffs = Vector{Float64}(undef, length(vec_edge_seq))
        Threads.@threads :greedy for i in eachindex(vec_edge_seq)
            subtr_msg_diffs[i] = norm(index_safe_message_subtract(new_messages[i], prev_messages[i]))
            dot_msg_diffs[i] = message_diff(new_messages[i], prev_messages[i])
        end
    end
    # mutate bpch object
    bpch.bpc = new_bpc
    bpch.message_history[idx] = new_messages
    bpch.residue_history[idx] = greedy_residues
    bpch.greedy_step_history[idx] = greedy_steps
    bpch.history_head = idx
    bpch.history_len = bpch.history_len < cap ? bpch.history_len + 1 : cap
    if return_msgdiff
        return res_norms, subtr_msg_diffs, dot_msg_diffs
    end
    return res_norms
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

function simultaneous_greedy_update(
    bpc::BeliefPropagationCache{V, N, M},
    es::AbstractVector{<:NamedEdge};
    update_alg = set_default_kwargs(Algorithm(default_message_update_alg(bpc)), bpc)
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    n = length(es)
    greedy_msgs = Vector{M}(undef, n) # use vector for thread safety
    greedy_residues = Vector{M}(undef, n) # use vector for thread safety
    # carry out contractions in prallel threads
    Threads.@threads :greedy for i in eachindex(es)
        e = es[i]
        new_message, _ = updated_message(update_alg, bpc, e)
        old_message = message(bpc, e)
        greedy_msgs[i] = new_message
        greedy_residues[i] = index_safe_message_subtract(new_message, old_message)
    end
    return greedy_msgs, greedy_residues
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

function anderson_acceleration_update(
        bpch::BeliefPropagationCacheHistory{V, N, M},
        memory_window::Int, # how many of the previous message updates should be used to construct the Anderson-accelerated update
        ;
        alphas = nothing,

    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}    
    
    n_prev = min(memory_window, history_length(bpch))
    past_greedys = Vector{Vector{M}}(undef, n_prev)
    prev_residuals = Vector{Vector{M}}(undef, n_prev)
    for i in 1:n_prev
        past_greedys[i] = past_greedy_steps(bpch, i)
        prev_residuals[i] = past_residues(bpch, i)
    end
    vec_edge_seq = vec_edge_sequence(bpch)
    n_edges = length(vec_edge_seq)
    # solve the least-squares problem to find optimal linear combination of the previous messages based on their respective residues
    alpha = anderson_least_squares(prev_residuals, vec_edge_seq)
    if !isnothing(alphas)
        push!(alphas, copy(alpha))
    end
    # construct the Anderson-accelerated message update and accept directly, although it may not be PD!
    AA_messages = Vector{M}(undef, n_edges)
    Threads.@threads :greedy for e_idx in 1:n_edges
        m_AA = sum(alpha[i] * past_greedys[i][e_idx] for i in 1:n_prev)
        m_AA = make_hermitian(m_AA)
        m_norm = tr(m_AA)
        AA_messages[e_idx] = m_AA / m_norm
    end
    return AA_messages
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

function update_with_anderson_acceleration(
        bpc::BeliefPropagationCache;
        memory_window = 10,
        maxiter = 100,
        alphas = nothing,
        avg_residual_norms = nothing,
        max_residual_norms = nothing,
        avg_subtr_msg_diffs = nothing,
        avg_dot_msg_diffs = nothing,
        check_PD = false,
        residual_tol::Float64 = default_residual_tol(),
        residual_diff_tol::Float64 = default_residual_diff_tol(),
        msgdiff_tol::Float64 = default_msgdiff_tol(),
    )
    vec_edge_seq = collect(edge_sequence(bpc))
    # initialize anderson acceleration with one greedy update step
    update_alg = set_default_kwargs(Algorithm(default_message_update_alg(bpc)), bpc)
    x0 = [message(bpc, e) for e in vec_edge_seq] # x_0 = initial messages
    u0, r0 = simultaneous_greedy_update(bpc, vec_edge_seq; update_alg = update_alg) # u_0 = one greedy update based on x_0, r_0 = greedy residuals u_0 - x_0
    x1 = u0 # for initialization, the accepted step is the greedy step
    bpc = update_bpc_messages(bpc, x1, vec_edge_seq)
    u1, r1 = simultaneous_greedy_update(bpc, vec_edge_seq; update_alg = update_alg)
    bpch = BeliefPropagationCacheHistory(bpc, x0, x1, u0, u1, r0, r1; history_capacity = memory_window)
    # start Anderson update loop
    for i in 2:maxiter
        AA_messages = anderson_acceleration_update(
            bpch,
            memory_window;
            alphas,
        )
        res_norms, subtr_msg_diffs, dot_msg_diffs = update_history!(bpch, AA_messages; return_msgdiff = true)
        # ----- monitoring -----
        avg_residual_norm = mean(res_norms)
        max_residual_norm = maximum(res_norms)
        avg_subtr_msg_diff = mean(subtr_msg_diffs)
        avg_dot_msg_diff = mean(dot_msg_diffs)
        if !isnothing(avg_residual_norms)
            push!(avg_residual_norms, avg_residual_norm)
        end
        if !isnothing(max_residual_norms)
            push!(max_residual_norms, max_residual_norm)
        end
        if !isnothing(avg_subtr_msg_diffs)
            push!(avg_subtr_msg_diffs, avg_subtr_msg_diff)
        end
        if !isnothing(avg_dot_msg_diffs)
            push!(avg_dot_msg_diffs, avg_dot_msg_diff)
        end
        if avg_residual_norm  <= residual_tol
            println("Converged after $i iterations (residuals approximately zero).")
            return bpch, true
        end
        check_PD_with_threading = false
        if check_PD && !messages_are_PD(AA_messages; threaded=check_PD_with_threading)
            @warn "Messages no longer PD after iteration $i."
            return bpch, false
        end
    end
end
