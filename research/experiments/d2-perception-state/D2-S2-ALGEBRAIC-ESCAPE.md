# D2-S2 algebraic escape falsification

This is a bounded negative investigation for the separate conformer frontend
that uses full-utterance per-feature normalization. It is not the pinned
VoiceChat frontend: VoiceChat uses Parakeet with `norm_per_feature=false`.

## Candidate

For one feature and frame:

```text
y_t = (z_t - mean_0:L) / sqrt(var_0:L + epsilon)
```

If `mean` and `var` were fixed, the affine transform could be folded into a
following linear operation. That would not remove the full-session
statistics, however; it would only relocate a fixed transform.

## Falsification

As a future frame changes `mean` or `var`, the historical normalized input
changes. The first subsampling module applies learned linear operations and
then nonlinear activation. Subsequent SiLU, attention, causal convolution,
normalization, residual, and projection paths depend on the resulting
historical activations. Correcting a previously emitted frame therefore
requires either historical activations or replay of the affected prefix.

No compact sufficient-statistics correction was found that preserves the
current normalized outputs through those nonlinear/stateful operations.

```text
Scope: NOT_DEC / genuine dependency (conformer normalization path)
Recommendation: SUPPRESS
Reason: De == full utterance for the current normalized frame;
        no removable candidate-domain gap exists.
VoiceChat relevance: not applicable; the pinned VoiceChat projector passes
                    norm_per_feature=false.
```

The broader replacement of repeated encoder history by bounded retained state
remains `DEC_EXTENDED_STATE`; this negative result does not reduce the value
of the D2-S1 encoder state or the D2-S2 raw-frontend frontier.
