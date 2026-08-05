

"""
    BPTraceGradientResult

Container returned by `bp_trace_residual_loss_and_gradient`.

Fields:
- `loss`: scalar `1/2 * sum_e ||U_e - m_e||_F^2`.
- `gradients`: `Dict(edge => Matrix)` containing the projected message gradient.
- `residuals`: `Dict(edge => Matrix)` with `q_e = U_e - m_e`.
- `updates`: `Dict(edge => Matrix)` with trace-normalized updates `U_e`.
- `raw_updates`: `Dict(edge => Matrix)` with raw updates `Y_e = \tilde U_e`.
- `output_adjoints`: `Dict(edge => Matrix)` with the adjoint inserted into the local reverse contraction.
- `edges`: directed edge list used for the computation.
"""
struct BPTraceGradientResult
    loss::Float64
    gradients::Dict{Any,Matrix{ComplexF64}}
    residuals::Dict{Any,Matrix{ComplexF64}}
    updates::Dict{Any,Matrix{ComplexF64}}
    raw_updates::Dict{Any,Matrix{ComplexF64}}
    output_adjoints::Dict{Any,Matrix{ComplexF64}}
    edges::Vector{Any}
end

# -----------------------------------------------------------------------------
# Small linear algebra helpers
# -----------------------------------------------------------------------------

_real_hs(A::AbstractMatrix, B::AbstractMatrix) = real(tr(adjoint(A) * B))
_fro2(A::AbstractMatrix) = _real_hs(A, A)

function _eye_complex(D::Integer)
    return Matrix{ComplexF64}(I, D, D)
end

function _hermitize(A::AbstractMatrix)
    H = Matrix{ComplexF64}(A)
    return (H + adjoint(H)) / 2
end

function _project_tracezero_hermitian(A::AbstractMatrix)
    H = _hermitize(A)
    D = size(H, 1)
    H .-= (tr(H) / D) .* _eye_complex(D)
    return _hermitize(H)
end

function _unique_preserve_order(xs)
    out = Any[]
    for x in xs
        if !any(y -> isequal(y, x), out)
            push!(out, x)
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Edge helpers. These are deliberately permissive because TNQS edge objects may
# be Graphs edges, NamedGraphs edges, Pair-like objects, or custom edge structs.
# -----------------------------------------------------------------------------

function _edge_src(e)
    for nm in (:src, :source, :from)
        if hasproperty(e, nm)
            return getproperty(e, nm)
        end
    end
    if e isa Pair
        return first(e)
    end
    if isdefined(@__MODULE__, :Graphs)
        try
            return Graphs.src(e)
        catch
        end
    end
    try
        return first(e)
    catch
    end
    error("Could not infer source vertex of edge $(repr(e)). Provide Pair-like edges or extend `_edge_src`.")
end

function _edge_dst(e)
    for nm in (:dst, :dest, :destination, :target, :to)
        if hasproperty(e, nm)
            return getproperty(e, nm)
        end
    end
    if e isa Pair
        return last(e)
    end
    if isdefined(@__MODULE__, :Graphs)
        try
            return Graphs.dst(e)
        catch
        end
    end
    try
        return last(e)
    catch
    end
    error("Could not infer destination vertex of edge $(repr(e)). Provide Pair-like edges or extend `_edge_dst`.")
end

function _default_edges(bpc)
    return _unique_preserve_order(collect(TN.edge_sequence(bpc)))
end

function _incoming_edges_to_vertex(directed_edges, v; exclude_source=nothing)
    out = Any[]
    for f in directed_edges
        if isequal(_edge_dst(f), v)
            if exclude_source === nothing || !isequal(_edge_src(f), exclude_source)
                push!(out, f)
            end
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Message matrix <-> ITensor conversion helpers
# -----------------------------------------------------------------------------

function _message_inds(bpc, e)
    dmsg = TN.default_message(TN.network(bpc), e)
    is = collect(inds(dmsg))
    length(is) == 2 || error("Expected rank-2 message for edge $(repr(e)), got $(length(is)) indices.")
    return is
end

function _message_matrix(bpc, e)
    msg = TN.message(bpc, e)
    is = _message_inds(bpc, e)
    return Matrix{ComplexF64}(Array(msg, is...))
end

function _itensor_from_message_matrix(bpc, e, M::AbstractMatrix)
    is = _message_inds(bpc, e)
    return ITensor(Matrix{ComplexF64}(M), is...)
end

function _set_message_matrix!(bpc, e, M::AbstractMatrix)
    TN.setmessage!(bpc, e, _itensor_from_message_matrix(bpc, e, M))
    return bpc
end

# -----------------------------------------------------------------------------
# Adapter: local factor extraction
# -----------------------------------------------------------------------------

"""
    default_local_bp_factor(bpc, vertex)

Return the local BP factor tensor at `vertex`, before inserting incoming
messages. The reverse contraction code needs this tensor.

This default adapter tries `TN.contract_bp_factors(bpc, vertex)` and then
`TN.contract_bp_factors(TN.network(bpc), vertex)`. If your TNQS version uses a
different internal/public function name, define your own

    my_local_factor(bpc, v) = ...

and pass `local_factor_fn = my_local_factor` to
`bp_trace_residual_loss_and_gradient`.
"""
function default_local_bp_factor(bpc, vertex)
    if isdefined(TN, :contract_bp_factors)
        f = getfield(TN, :contract_bp_factors)
        try
            return f(bpc, vertex)
        catch err1
            try
                return f(TN.network(bpc), vertex)
            catch err2
                error("TN.contract_bp_factors exists, but both calls failed. First error: $(err1). Second error: $(err2). Pass `local_factor_fn` explicitly.")
            end
        end
    end
    error("Could not find `TN.contract_bp_factors`. Pass `local_factor_fn = (bpc, v) -> ...` that returns the local BP factor tensor.")
end

# -----------------------------------------------------------------------------
# Forward raw update and reverse local VJP
# -----------------------------------------------------------------------------

function _raw_update_tnqs(bpc, e; sequence_alg="optimal")
    alg_raw = TN.Algorithm("contract"; normalize=false, sequence_alg=sequence_alg)
    raw, _ = TN.updated_message(alg_raw, bpc, e)
    return raw
end

function _raw_update_local(bpc, e; directed_edges, local_factor_fn)
    i = _edge_src(e)
    j = _edge_dst(e)
    Y = local_factor_fn(bpc, i)
    for f in _incoming_edges_to_vertex(directed_edges, i; exclude_source=j)
        Y *= TN.message(bpc, f)
    end
    return Y
end

"""
    check_local_raw_update(bpc, e; kwargs...)

Compare the raw update from TNQS `updated_message(...; normalize=false)` with
the raw update obtained by contracting `local_factor_fn(bpc, src(e))` with the
incoming messages. This is a useful adapter/index-convention check.
"""
function check_local_raw_update(
    bpc,
    e;
    directed_edges=_default_edges(bpc),
    local_factor_fn=default_local_bp_factor,
    sequence_alg="optimal",
    hermitize=false,
)
    is = _message_inds(bpc, e)
    raw_tn = _raw_update_tnqs(bpc, e; sequence_alg=sequence_alg)
    raw_lc = _raw_update_local(bpc, e; directed_edges=directed_edges, local_factor_fn=local_factor_fn)
    A = Matrix{ComplexF64}(Array(raw_tn, is...))
    B = Matrix{ComplexF64}(Array(raw_lc, is...))
    if hermitize
        A = _hermitize(A)
        B = _hermitize(B)
    end
    denom = max(norm(A), norm(B), eps(Float64))
    return (; absdiff=norm(A - B), reldiff=norm(A - B) / denom, tnqs=A, local=B)
end

function _add_reverse_local_contributions!(
    gradients::Dict{Any,Matrix{ComplexF64}},
    bpc,
    e,
    output_adjoint::AbstractMatrix;
    directed_edges,
    local_factor_fn,
    conjugate_local_network::Bool=true,
)
    i = _edge_src(e)
    j = _edge_dst(e)
    inputs = _incoming_edges_to_vertex(directed_edges, i; exclude_source=j)

    # Coefficient-wise conjugation implements the adjoint of the multilinear
    # local contraction under the real Frobenius/Hilbert-Schmidt inner product.
    # It does not transpose the two message legs. We project to Hermitian
    # trace-zero directions after accumulation.
    T = local_factor_fn(bpc, i)
    Tadj = conjugate_local_network ? conj(T) : T
    Ybar_it = _itensor_from_message_matrix(bpc, e, output_adjoint)

    for f in inputs
        Z = Tadj * Ybar_it
        for g in inputs
            isequal(g, f) && continue
            msg_g = TN.message(bpc, g)
            Z *= conjugate_local_network ? conj(msg_g) : msg_g
        end
        isf = _message_inds(bpc, f)
        Gf = Matrix{ComplexF64}(Array(Z, isf...))
        gradients[f] .+= Gf
    end
    return gradients
end

# -----------------------------------------------------------------------------
# Main public functions
# -----------------------------------------------------------------------------

"""
    bp_trace_residual_loss_and_gradient(bpc; kwargs...) -> BPTraceGradientResult

Compute the trace-gauge BP residual loss and the direct gradient
`J_q^\dagger q` with respect to the messages, without finite differences and
without explicitly forming `J_q`.

The residual is

    q_e = U_e - m_e,
    U_e = raw_e / tr(raw_e),
    raw_e = \tilde U_e(m).

For each residual, the output adjoint inserted into the reverse local
contraction is

    Ybar_e = (q_e - <q_e, U_e> I) / tr(raw_e),

where `<A,B> = real(tr(A'B))`.

Keyword arguments:
- `directed_edges`: directed edges to include; defaults to unique `TN.edge_sequence(bpc)`.
- `local_factor_fn`: function `(bpc, vertex) -> ITensor` returning the local BP factor.
- `sequence_alg`: sequence algorithm passed to TNQS raw forward update.
- `forward_raw_update`: `:tnqs` uses `TN.updated_message`; `:local` uses local factor contraction.
- `hermitize_messages`: use Hermitian part of current messages in the residual.
- `hermitize_updates`: use Hermitian part of normalized updates in the residual.
- `project_gradient`: project final gradients to Hermitian trace-zero matrices.
- `conjugate_local_network`: use coefficient-wise conjugation in reverse contractions.

Returns matrices indexed by edge in `result.gradients`.
"""
function bp_trace_residual_loss_and_gradient(
    bpc;
    directed_edges=_default_edges(bpc),
    local_factor_fn=default_local_bp_factor,
    sequence_alg="optimal",
    forward_raw_update::Symbol=:tnqs,
    hermitize_messages::Bool=true,
    hermitize_updates::Bool=true,
    project_gradient::Bool=true,
    conjugate_local_network::Bool=true,
    check_finite_traces::Bool=true,
)
    edges = _unique_preserve_order(collect(directed_edges))

    gradients = Dict{Any,Matrix{ComplexF64}}()
    residuals = Dict{Any,Matrix{ComplexF64}}()
    updates = Dict{Any,Matrix{ComplexF64}}()
    raw_updates = Dict{Any,Matrix{ComplexF64}}()
    output_adjoints = Dict{Any,Matrix{ComplexF64}}()

    # Allocate gradient buffers.
    for e in edges
        M = _message_matrix(bpc, e)
        gradients[e] = zeros(ComplexF64, size(M, 1), size(M, 2))
    end

    loss = 0.0

    # Forward pass: raw updates, trace normalization, residuals, output adjoints.
    for e in edges
        is = _message_inds(bpc, e)

        raw_it = if forward_raw_update === :tnqs
            _raw_update_tnqs(bpc, e; sequence_alg=sequence_alg)
        elseif forward_raw_update === :local
            _raw_update_local(bpc, e; directed_edges=edges, local_factor_fn=local_factor_fn)
        else
            error("forward_raw_update must be :tnqs or :local, got $(repr(forward_raw_update)).")
        end

        Y = Matrix{ComplexF64}(Array(raw_it, is...))
        Y_use = hermitize_updates ? _hermitize(Y) : Y

        s = real(tr(Y_use))
        if check_finite_traces && (!isfinite(s) || abs(s) <= sqrt(eps(Float64)))
            error("Bad trace normalizer for edge $(repr(e)): tr(raw) = $s")
        end

        U = Y_use ./ s
        U = hermitize_updates ? _hermitize(U) : U

        M = _message_matrix(bpc, e)
        M = hermitize_messages ? _hermitize(M) : M
        # The algorithm assumes trace-one messages. We do not silently normalize
        # here, because doing so would make the returned gradient inconsistent
        # with the cache. Normalize the cache before calling this routine.

        Q = U - M
        if hermitize_updates || hermitize_messages
            Q = _hermitize(Q)
        end

        loss += 0.5 * _fro2(Q)

        residuals[e] = Q
        updates[e] = U
        raw_updates[e] = Y

        # Direct residual dependence on m_e: q_e = U_e - m_e.
        gradients[e] .-= Q

        D = size(Q, 1)
        Id = _eye_complex(D)
        coeff = _real_hs(Q, U)
        Ybar = (Q .- coeff .* Id) ./ s
        output_adjoints[e] = Ybar
    end

    # Reverse pass: for each residual, push its output adjoint through the local
    # BP contraction and accumulate contributions on its incoming messages.
    for e in edges
        _add_reverse_local_contributions!(
            gradients,
            bpc,
            e,
            output_adjoints[e];
            directed_edges=edges,
            local_factor_fn=local_factor_fn,
            conjugate_local_network=conjugate_local_network,
        )
    end

    if project_gradient
        for e in edges
            gradients[e] = _project_tracezero_hermitian(gradients[e])
        end
    else
        for e in edges
            gradients[e] = _hermitize(gradients[e])
        end
    end

    return BPTraceGradientResult(loss, gradients, residuals, updates, raw_updates, output_adjoints, edges)
end

"""
    loss_only_trace_residual(bpc; kwargs...) -> Float64

Compute only `1/2 * sum_e ||U_e - m_e||_F^2`, using the same trace-gauge
residual convention as `bp_trace_residual_loss_and_gradient`.
"""
function loss_only_trace_residual(
    bpc;
    directed_edges=_default_edges(bpc),
    sequence_alg="optimal",
    forward_raw_update::Symbol=:tnqs,
    local_factor_fn=default_local_bp_factor,
    hermitize_messages::Bool=true,
    hermitize_updates::Bool=true,
)
    edges = _unique_preserve_order(collect(directed_edges))
    loss = 0.0
    for e in edges
        is = _message_inds(bpc, e)
        raw_it = if forward_raw_update === :tnqs
            _raw_update_tnqs(bpc, e; sequence_alg=sequence_alg)
        elseif forward_raw_update === :local
            _raw_update_local(bpc, e; directed_edges=edges, local_factor_fn=local_factor_fn)
        else
            error("forward_raw_update must be :tnqs or :local, got $(repr(forward_raw_update)).")
        end
        Y = Matrix{ComplexF64}(Array(raw_it, is...))
        Y = hermitize_updates ? _hermitize(Y) : Y
        U = Y ./ real(tr(Y))
        U = hermitize_updates ? _hermitize(U) : U
        M = _message_matrix(bpc, e)
        M = hermitize_messages ? _hermitize(M) : M
        Q = U - M
        Q = (hermitize_updates || hermitize_messages) ? _hermitize(Q) : Q
        loss += 0.5 * _fro2(Q)
    end
    return loss
end

"""
    additive_gradient_step!(bpc, result; eta, edges=result.edges, renormalize_trace=true)

Apply a simple additive projected-gradient step to the messages stored in `bpc`:

    m_e <- m_e - eta * result.gradients[e].

The result is hermitized and trace-normalized. This routine is meant as a
minimal reference update rule; for serious optimization, pass `result.loss` and
`result.gradients` to an optimizer/line search/L-BFGS driver.
"""
function additive_gradient_step!(
    bpc,
    result::BPTraceGradientResult;
    eta::Real,
    edges=result.edges,
    renormalize_trace::Bool=true,
    hermitize::Bool=true,
)
    for e in edges
        M = _message_matrix(bpc, e)
        G = result.gradients[e]
        Mnew = M .- eta .* G
        if hermitize
            Mnew = _hermitize(Mnew)
        end
        if renormalize_trace
            Mnew ./= real(tr(Mnew))
        end
        _set_message_matrix!(bpc, e, Mnew)
    end
    return bpc
end

end # module
