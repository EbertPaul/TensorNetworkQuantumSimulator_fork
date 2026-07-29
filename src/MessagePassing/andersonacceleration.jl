using ITensors: Algorithm
using Dictionaries: Dictionary, set!

struct BeliefPropagationCacheHistory{V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    current_BeliefPropagationCache::BeliefPropagationCache{V, N, M}      
    message_history::Vector{Dictionary{NamedEdge, M}}                   # messages that were accepted at different iterations of the message update process
    message_proposal_history::Vector{Dictionary{NamedEdge, M}}          # messages that were proposed by the greedy update scheme at different iterations of the message update process
    message_residues_history::Vector{Dictionary{NamedEdge, M}}          # difference between the proposed greedy message update at iteration k and the accepted message at iteration k-1 (zero residue = BP fixed point)
end

function BeliefPropagationCacheHistory(bpc::BeliefPropagationCache{V, N, M}) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    return BeliefPropagationCacheHistory(
        bpc,                   
        Dictionary{NamedEdge, M}[],
        Dictionary{NamedEdge, M}[], # cannot be known without information about how the current messages were obtained
        Dictionary{NamedEdge, M}[], # cannot be known without information about how the current messages were obtained
        )
end

function BeliefPropagationCache(bpch::BeliefPropagationCacheHistory)
    return bpch.current_BeliefPropagationCache
end

function history_length(bpch::BeliefPropagationCacheHistory)
    ml = length(bpch.message_history)
    pl = length(bpch.message_proposal_history)
    rl = length(bpch.message_residues_history)
    if ml != pl || ml != rl
        error("The lengths of the message history, message proposal history, and message residues history are inconsistent! Got lengths: message_history=$ml, message_proposal_history=$pl, message_residues_history=$rl")
    end
    return ml
end

function past_messages(bpch::BeliefPropagationCacheHistory, iterations_into_past::Int)
    if iterations_into_past < 0
        error("When accessing prior accepted messages iterations_into_past must be a non-negative integer, where 0 corresponds to the current state of the BeliefPropagationCache.")
    end
    if iterations_into_past == 0
        return messages(bpch.current_BeliefPropagationCache)
    elseif iterations_into_past <= history_length(bpch)
        return bpch.message_history[end - iterations_into_past + 1]
    else
        error("The stored message history contains $(history_length(bpch)) iterations, but you requested a message from $(iterations_into_past) iterations into the past.")
    end
end

function past_message_proposals(bpch::BeliefPropagationCacheHistory, iterations_into_past::Int)
    if iterations_into_past <= 0
        error("When accessing prior message proposals iterations_into_past must be a positive integer, where 1 corresponds to the most recent proposal that lead to the current state of the BeliefPropagationCache.")
    end
    if iterations_into_past <= history_length(bpch)
        return bpch.message_proposal_history[end - iterations_into_past + 1]
    else
        error("The stored message proposal history contains $(history_length(bpch)) iterations, but you requested a message proposal from $(iterations_into_past) iterations into the past.")
    end
end

function past_message_residues(bpch::BeliefPropagationCacheHistory, iterations_into_past::Int)
    if iterations_into_past <= 0
        error("When accessing prior message residues iterations_into_past must be a positive integer, where 1 corresponds to the most recent residue that lead to the current state of the BeliefPropagationCache.")
    end
    if iterations_into_past <= history_length(bpch)
        return bpch.message_residues_history[end - iterations_into_past + 1]
    else
        error("The stored message residues history contains $(history_length(bpch)) iterations, but you requested a message residue from $(iterations_into_past) iterations into the past.")
    end
end

function update_history(bpch::BeliefPropagationCacheHistory,
        new_bpc::BeliefPropagationCache,
        new_message_proposals::Dictionary{NamedEdge, M},
        new_message_residues::Dictionary{NamedEdge, M}) where M <: Union{ITensor, Vector{ITensor}}
    
    new_message_history = copy(bpch.message_history)
    push!(new_message_history, copy(messages(new_bpc)))
    new_message_proposal_history = copy(bpch.message_proposal_history)
    push!(new_message_proposal_history, new_message_proposals)
    new_message_residues_history = copy(bpch.message_residues_history)
    push!(new_message_residues_history, new_message_residues)
    return BeliefPropagationCacheHistory(
        new_bpc,
        new_message_history,
        new_message_proposal_history,
        new_message_residues_history
    )
end

function update_bpc(bpc::BeliefPropagationCache, new_messages::Dictionary{NamedEdge, M}) where M <: Union{ITensor, Vector{ITensor}}
    return BeliefPropagationCache(
        network(bpc),
        new_messages,
        contraction_sequences(bpc),
        edge_sequence(bpc)
    )
end

function update_history(
    bpch::BeliefPropagationCacheHistory,
    new_accepted_messages::Dictionary{NamedEdge, M},
    new_message_proposals::Dictionary{NamedEdge, M},
    new_message_residues::Dictionary{NamedEdge, M}) where M <: Union{ITensor, Vector{ITensor}}
    new_bpc = update_bpc(bpch.current_BeliefPropagationCache, new_accepted_messages)
    return update_history(bpch, new_bpc, new_message_proposals, new_message_residues)
end

function simultaneous_greedy_update(bpc::BeliefPropagationCache{V, N, M}) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}
    message_proposals = Dictionary{NamedEdge, M}()
    update_alg = set_default_kwargs(Algorithm(default_message_update_alg(bpc)), bpc)
    for e in edge_sequence(bpc)
        m, (cache_key, sequence, seq_changed) = updated_message(update_alg, bpc, e)
        set!(message_proposals, e, m)
    end
    message_residues = Dictionary{NamedEdge, M}()
    for e in edge_sequence(bpc)
        set!(message_residues, e, message_proposals[e] - message(bpc, e))
    end
    return message_proposals, message_residues
end

function anderson_least_squares(
        residues::AbstractVector{<:Dictionary{NamedEdge, Union{ITensor, Vector{ITensor}}}},
        edges::AbstractVector{<:NamedEdge},
    )
    N = length(residues) # M + 1 = number of message proposals from the past + the current one
    C = zeros(N + 1, N + 1)
    for i in 1:N
        for j in i:N
            Cij = 0.0
            for e in edges
                Cij += real(scalar(dag(residues[i][e]) * residues[j][e]))
            end
            C[i, j] = Cij
            C[j, i] = Cij
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

function anderson_accelerated_update_step(
        bpch::BeliefPropagationCacheHistory{V, N, M},
        memory_window::Int, # how many of the previous message updates should be used to construct the Anderson-accelerated update
        #msg_eigenvalue_lower_bound::Dictionary{NamedEdge, Float64}, # lower bound on the eigenvalues of each message
        ;
        alphas = nothing,
    ) where {V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{ITensor}}}

    current_messages = messages(bpch.current_BeliefPropagationCache)
    new_message_proposals, new_message_residues = simultaneous_greedy_update(bpch.current_BeliefPropagationCache)
    n_prev = min(memory_window, history_length(bpch)) # number of previous update steps that are taken into account
    proposals = [new_message_proposals]
    residues = [new_message_residues]
    for i in 1:n_prev
        push!(proposals, past_message_proposals(bpch, i))
        push!(residues, past_message_residues(bpch, i))
    end

    # solve the least-squares problem to find optimal linear combination of the current and the M previous message proposals
    edges = edge_sequence(bpch.current_BeliefPropagationCache)
    alpha = anderson_least_squares(residues, edges)
    if !isnothing(alphas)
        push!(alphas, copy(alpha))
    end

    # construct the Anderson-accelerated message update
    anderson_message_proposals = Dictionary{NamedEdge, M}()
    for e in edges
        m_AA = sum(alpha[i] * proposals[i][e] for i in 1:(n_prev + 1))
        m_AA = make_hermitian(m_AA)
        m_norm = tr(m_AA)
        if iszero(m_norm)
            set!(anderson_message_proposals, e, current_messages[e])
        else
            set!(anderson_message_proposals, e, m_AA / m_norm)
        end
    end

    # take m_AA as the actual accepted messages although they may not be positive definite
    new_accepted_messages = Dictionary{NamedEdge, M}()
    for e in edges
        set!(new_accepted_messages, e, anderson_message_proposals[e])
    end
    #=

    # Anderson-accelerated message need not to be positive definite (entries of alpha can be negative!), so implement safeguard mechanism
    # -> find beta for which the update m_e + beta * (m_AA - m_e) is positive definite for all edges e
    beta = 1
    theta = 0.8 # safeguard parameter between 0 (useless) and 1 (most risky)
    D_e_Frob_dict = Dictionary{NamedEdge, Float64}()
    for e in edges
        delta_e = msg_eigenvalue_lower_bound[e]
        D_e = anderson_message_proposals[e] - current_messages[e]
        D_e_Frob = sqrt(real(scalar(dag(D_e) * D_e)))
        beta_e = iszero(D_e_Frob) ? 1.0 : theta * delta_e / D_e_Frob
        beta = min(beta, beta_e)
        set!(D_e_Frob_dict, e, D_e_Frob)
    end

    # contruct the actual message update with PD safeguard taken into account
    new_accepted_messages = Dictionary{NamedEdge, M}()
    for e in edges
        set!(new_accepted_messages, e, current_messages[e] + beta * (anderson_message_proposals[e] - current_messages[e]))
    end

    # update the msg_engenvalue_lower_bound for the next iteration
    new_msg_eigenvalue_lower_bound = Dictionary{NamedEdge, Float64}()
    for e in edges
        set!(new_msg_eigenvalue_lower_bound, e, msg_eigenvalue_lower_bound[e] - beta * D_e_Frob_dict[e])
        if new_msg_eigenvalue_lower_bound[e] < 0
            @warn "The lower bound on the eigenvalue of the message along edge $e has become negative: $(new_msg_eigenvalue_lower_bound[e]). This should not happen under any circumstances!"
        end
    end
    =#

    return new_accepted_messages, new_message_proposals, new_message_residues
end

function residuals_approx_zero(residuals::Dictionary{NamedEdge, M}; tol::Float64 = 1e-8) where M <: Union{ITensor, Vector{ITensor}}
    for e in keys(residuals)
        if norm(residuals[e]) > tol
            return false
        end
    end
    return true
end

function residual_diff_approx_zero(residuals_a::Dictionary{NamedEdge, M}, residuals_b::Dictionary{NamedEdge, M}; tol::Float64 = 1e-8) where M <: Union{ITensor, Vector{ITensor}}
    for e in keys(residuals_a)
        if norm(residuals_a[e] - residuals_b[e]) > tol
            return false
        end
    end
    return true
end

function msgdiff_approx_zero(msgs_a::Dictionary{NamedEdge, M}, msgs_b::Dictionary{NamedEdge, M}; tol::Float64 = 1e-8) where M <: Union{ITensor, Vector{ITensor}}
    for e in keys(msgs_a)
        if message_diff(msgs_a[e], msgs_b[e]) > tol
            return false
        end
    end
    return true
end

function average_residual_norm(residuals::Dictionary{NamedEdge, M}) where M <: Union{ITensor, Vector{ITensor}}
    total_norm = 0.0
    for e in keys(residuals)
        total_norm += norm(residuals[e])
    end
    return total_norm / length(keys(residuals))
end

function average_message_diff(msgs_a::Dictionary{NamedEdge, M}, msgs_b::Dictionary{NamedEdge, M}) where M <: Union{ITensor, Vector{ITensor}}
    total_diff = 0.0
    for e in keys(msgs_a)
        total_diff += message_diff(msgs_a[e], msgs_b[e])
    end
    return total_diff / length(keys(msgs_a))
end

function messages_are_PD(messages, edges)
    for e in edges
        m = messages[e]
        inds = collect(ITensors.inds(m))
        M =  Array(m, inds...)
        if !isposdef(M)
            return false
        end
    end
    return true
end




function update_with_anderson_acceleration(
        network;
        memory_window = 5,
        maxiter = 20,
        alphas = nothing,
        avg_residual_norms = nothing,
        avg_msg_diffs = nothing,
        check_PD = false
    )
    # initialize the BeliefPropagationCache and the lower bound on the eigenvalues for each message
    bpc = BeliefPropagationCache(network) 
    msg_eigenvalue_lower_bound = Dictionary{NamedEdge, Float64}()
    edges = edge_sequence(bpc)
    for e in edges
        m_e = message(bpc, e)
        set!(bpc.messages, e, m_e) # enforces message initialization
        inds = collect(ITensors.inds(m_e))
        M =  Array(m_e, inds...)
        # assuming that m_e = delta up to a normalization
        set!(msg_eigenvalue_lower_bound, e, real(M[1, 1]))
    end

    # initialize the BeliefPropagationCacheHistory
    bpch = BeliefPropagationCacheHistory(bpc)

    # start performing Anderson-accelerated message updates and check for convergence or stagnation
    for i in 1:maxiter
        #println("Iteration $i")
        prev_messages = copy(messages(bpch.current_BeliefPropagationCache))
        prev_residuals = i > 1 ? past_message_residues(bpch, 1) : nothing
        new_accepted_messages, new_message_proposals, new_message_residues = anderson_accelerated_update_step(
            bpch,
            memory_window;
            alphas,
        )
        bpch = update_history(bpch, new_accepted_messages, new_message_proposals, new_message_residues)
        
        # report to user
        avg_residual_norm = average_residual_norm(new_message_residues)
        avg_msg_diff = average_message_diff(new_accepted_messages, prev_messages)
        if !isnothing(avg_residual_norms)
            push!(avg_residual_norms, avg_residual_norm)
        end
        if !isnothing(avg_msg_diffs)
            push!(avg_msg_diffs, avg_msg_diff)
        end
        
        # residuals = 0 implies BP fixed point -> test convergence
        res_tol = 1e-10
        if residuals_approx_zero(new_message_residues; tol=res_tol)
            println("Converged after $i iterations (residuals approximately zero).")
            return bpch.current_BeliefPropagationCache
        end
        # check for stagnation based on messages (e.g. because the step size becomes too small)
        msgdiff_tol = 1e-14
        if msgdiff_approx_zero(new_accepted_messages, prev_messages; tol=msgdiff_tol)
            println("Stagnated after $i iterations (message approximately unchanged).")
            return bpch.current_BeliefPropagationCache
        end
        # check for stagnation based on residuals (e.g. because the step size becomes too small)
        res_diff_tol = 1e-14
        if !isnothing(prev_residuals) && residual_diff_approx_zero(new_message_residues, prev_residuals; tol=res_diff_tol)
            println("Stagnated after $i iterations (residual difference approximately zero).")
            return bpch.current_BeliefPropagationCache
        end

        if check_PD && !messages_are_PD(new_accepted_messages, edges)
            error("Messages no longer PD after iteration $i.")
        end
    end
    println("Reached maximum number of iterations ($maxiter) without convergence or stagnation.")
    return bpch.current_BeliefPropagationCache
end











