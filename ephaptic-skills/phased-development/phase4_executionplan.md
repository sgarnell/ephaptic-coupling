

**1. Restatement Of Goals For Phase 4**

Phase 3 is a layer-wise global pulse. Phase 4 is a spatiotemporal propagation model. The key architectural change is that brightness stops being a single scalar per frame and becomes a per-point field evaluated over space and time.

- In the current Phase 3 implementation in animation.py, `generate_frame(plotter, t, config)` computes one scalar:
  $$
  B(t)=1+\mathrm{A}\sin(2\pi f t)
  $$
  where `A = config.amplitude` and `f = config.frequency`.
- That scalar is applied once per layer by converting `base_rgb -> HSV`, scaling the value channel `v`, clamping to `[0,1]`, converting back to RGB, and then tiling the same RGB over every point in `mesh.point_data["colors"]`.
- The result is synchronized pulsing. All points in a given layer brighten and dim together. All layers also share the same temporal phase.
- Phase 4, per [ephaptic-skills/phased-development/phase4.md](ephaptic-skills/phased-development/phase4.md), changes the model to:
  $$
  B(\mathbf{x}, t)=1+\mathrm{A}\sin(\mathbf{k}\cdot\mathbf{x}-\omega t+\phi)
  $$
- In Phase 4, two points with different positions $\mathbf{x}$ can have different brightness at the same time $t$, which produces visible traveling wavefronts rather than a global pulse.
- Phase 3 is temporal-only modulation. Phase 4 is temporal plus spatial modulation.
- Phase 3 uses one brightness value per frame. Phase 4 uses one brightness value per point per frame.
- Phase 4 must still preserve the current rendering contract in animation.py: do not rebuild geometry, only mutate `mesh.point_data["colors"]` and render.

**2. Required Additions To AnimationConfig**
The current `AnimationConfig` in animation.py has `duration`, `fps`, `amplitude`, `frequency`, and `color_map`. Phase 4 needs explicit, typed fields added to that dataclass.

- `mode: str = "global_pulse"`
  Purpose: selects the computation branch inside `generate_frame`.
  Interaction: preserves backward compatibility because existing code paths continue to work when `mode == "global_pulse"`.
- `wave_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)`
  Purpose: defines the propagation direction for the traveling wave.
  Interaction: used only when `mode == "traveling_wave"`. Ignored in `global_pulse`.
- `wavelength: float = 100.0`
  Purpose: spatial period $\lambda$ in the same coordinate units as the synapse points loaded by `get_layer_points`.
  Interaction: used with `wave_direction` to derive $\mathbf{k}$.
- `velocity: float = 1.0`
  Purpose: propagation speed along the wave direction.
  Interaction: used with $\|\mathbf{k}\|$ to derive $\omega$.
- `phase_offset: float = 0.0`
  Purpose: global phase shift $\phi$ applied to all points and layers.
  Interaction: additive with any optional per-layer delay term.
- `layer_coupling: float = 0.0`
  Purpose: optional v1 scalar controlling per-layer delay, without introducing a full coupling matrix.
  Interaction: ignored in `global_pulse`; in `traveling_wave`, converted into a simple delay or phase shift per layer.

How these interact with existing fields:

- `duration` and `fps` remain unchanged. They still control total frames and frame timing in `play_animation` and `export_animation`.
- `amplitude` remains the modulation depth in both modes.
- `frequency` remains authoritative only for `mode == "global_pulse"`.
- In `mode == "traveling_wave"`, `frequency` should not drive the wave. The temporal rate is instead implied by `velocity` and `wavelength` through $\omega = \|\mathbf{k}\|v`.
- `color_map` remains reserved and should stay behaviorally inert unless a later phase introduces hue remapping.

Mode switching mechanism:

- Default `mode` must be `"global_pulse"` so all existing tests and callers remain valid.
- `generate_frame` should branch on `config.mode`.
- `play_animation` and `export_animation` should remain signature-stable and simply pass the expanded config through.
- Any unsupported `mode` should fail fast with a `ValueError` from `generate_frame`, not silently fall back.

**3. Data Structures Needed For Phase 4**
The current private scene state is `plotter._anim_meshes`, a list of `(mesh, base_rgb)` tuples in [visualization/animation.py](visualization/animation.py). That is enough for Phase 3, but insufficient for Phase 4 because per-point computation needs cached spatial terms.

Required internal data:

- Per-point projected coordinate array `s = \mathbf{k}\cdot\mathbf{x}` for each mesh.
- Precomputed wave vector `k`, shape `(3,)`.
- Precomputed angular frequency `omega`, scalar.
- Optional per-layer delay value or per-layer delay array.
- Per-layer base HSV or equivalent cached scalars.

Recommended storage locations, without changing public APIs:

- Shared wave parameters should live on the `plotter` as private attributes, because they are common across all meshes in the current scene.
  Examples of what belongs there conceptually:
  `plotter._wave_k`, `plotter._wave_omega`, `plotter._wave_mode`.
- Per-mesh projected coordinates should live with each mesh record inside the existing private animation cache, not in `viewer.py`.
- The cleanest internal evolution is to replace the current `(mesh, base_rgb)` tuple with a richer private record in [visualization/animation.py](visualization/animation.py). It can still remain private and local to that module.
- Each per-mesh record should contain:
  `mesh`
  `base_rgb`
  `base_hsv` or at least `base_v`
  `projected_s`
  `layer_delay`
  `layer_num`
- If you want to avoid private dataclasses entirely, the alternative is to store `projected_s` in `mesh.point_data` as an auxiliary array and keep the rest in the existing `plotter._anim_meshes` record. That also preserves geometry and keeps caching local to the mesh.
- Do not move any of this into [visualization/viewer.py](visualization/viewer.py). That module should stay a static scene/view concern.

**4. Mathematical Model And Computation Pipeline**
Phase 4 should be implemented as a fully vectorized transformation from point coordinates to brightness factors to RGB arrays.

Wave parameter derivation:

- Normalize the configured direction:
  $$
  \hat{d}=\frac{\mathbf{d}}{\|\mathbf{d}\|}
  $$
- Derive the wave vector:
  $$
  \mathbf{k}=\frac{2\pi}{\lambda}\hat{d}
  $$
- Derive angular frequency from propagation speed:
  $$
  \omega=\|\mathbf{k}\|v = \frac{2\pi}{\lambda}v
  $$

Projected coordinate computation:

- For a mesh with `points` of shape `(N, 3)`, compute:
  $$
  s = \mathbf{X}\mathbf{k}
  $$
  where $\mathbf{X}$ is the `(N,3)` point matrix and $s$ is shape `(N,)`.
- In NumPy terms, that is a single vectorized dot product over all points, not a point loop.
- This `s` should be precomputed once when the animation scene is built or when the mode changes, not recomputed every frame.

Brightness evaluation:

- Base traveling-wave model:
  $$
  B(\mathbf{x}, t)=1 + A\sin(s-\omega t+\phi)
  $$
- With optional layer delay:
  $$
  B_\ell(\mathbf{x}, t)=1 + A\sin(s-\omega (t-\Delta t_\ell)+\phi)
  $$
  which can be rewritten as
  $$
  B_\ell(\mathbf{x}, t)=1 + A\sin(s-\omega t+\phi+\omega\Delta t_\ell)
  $$

HSV pipeline integration:

- The current Phase 3 path in animation.py is:
  `base_rgb -> hsv -> modify v -> hsv_to_rgb`.
- For Phase 4, doing that with `colorsys` per point would be too slow because `colorsys` is scalar Python code.
- The better engineering choice is to preserve the same math while avoiding per-point HSV conversion.
- For a fixed layer color, hue and saturation are constant, so changing only the HSV value channel is equivalent to scaling the layer RGB vector by a scalar derived from the new value.
- Let `base_rgb` correspond to a base HSV value `base_v`. Then:
  $$
  v_{\text{new}} = \mathrm{clip}(base_v \cdot B, 0, 1)
  $$
  and the RGB result can be computed as:
  $$
  rgb_{\text{new}} = base\_rgb \cdot \frac{v_{\text{new}}}{base_v}
  $$
  as long as `base_v > 0`.
- This keeps the behavior mathematically aligned with the Phase 3 HSV contract while making the Phase 4 path vectorizable.
- Because `get_layer_color` currently returns non-black colors, `base_v` should be positive in practice. Still, the implementation should guard the `base_v == 0` case and emit zeros if it ever occurs.

**5. Integration With generate_frame()**
The central architectural rule from [ephaptic-skills/phased-development/phase4.md](ephaptic-skills/phased-development/phase4.md) is correct: `generate_frame(plotter, t, config)` remains the dispatch point.

Exact dispatch logic:

- If `config.mode == "global_pulse"`:
  execute the existing Phase 3 behavior unchanged.
- If `config.mode == "traveling_wave"`:
  execute the new per-point traveling-wave branch.
- Else:
  raise a `ValueError` naming the unsupported mode.

How to preserve Phase 3 behavior:

- Keep the current formula:
  $$
  1 + config.amplitude \cdot \sin(2\pi \cdot config.frequency \cdot t)
  $$
- Keep the current semantics of updating all points in a mesh to the same color.
- Keep `plotter.render()` at the end of the function.
- Keep geometry untouched and continue calling `mesh.Modified()` after color changes.

How to integrate per-point brightness:

- For each cached mesh record, read its precomputed `projected_s`.
- Evaluate the brightness vector for that mesh:
  $$
  brightness = 1 + A\sin(projected\_s - \omega t + \phi + phase\_layer)
  $$
- Convert that brightness vector into a per-point RGB array using the precomputed layer base color and base value.
- Write the result directly into `mesh.point_data["colors"]`.

Efficient color-array update strategy:

- The current Phase 3 path allocates a new tiled array each frame. That is acceptable for one RGB per layer, but Phase 4 will produce an `(N,3)` array anyway.
- Prefer stable shape and dtype, ideally `float32`, because the current meshes are already `float32`.
- If practical, reuse the existing `mesh.point_data["colors"]` array and assign into it in place instead of replacing it with a new array object every frame.
- If PyVista/VTK update propagation proves unreliable with in-place assignment alone, assign the full array back once per frame and call `mesh.Modified()`. Correctness matters more than micro-optimizing the first implementation.

Avoiding geometry rebuilds:

- `_build_animation_scene` in [visualization/animation.py](visualization/animation.py) already sets up meshes once and adds them to the plotter.
- Phase 4 should only add cached per-mesh/per-plotter metadata there.
- `play_animation` and `export_animation` should still just build once, then call `generate_frame` repeatedly.

**6. Performance Strategy**
Phase 4 can remain efficient if all expensive spatial terms are cached and all per-frame work is vectorized.

Vectorization strategy:

- Precompute `k` once per config or once per built scene.
- Precompute `omega` once per config or once per built scene.
- Precompute `projected_s = points @ k` once per mesh.
- At frame time, use one NumPy `sin` call per mesh over the whole vector of points.
- Use vectorized `clip`, multiplication, and broadcasted RGB scaling over `(N,3)`.

Precomputation strategy:

- `wave_direction` normalization, `k`, and `omega` should be computed before the first frame, not inside the frame loop.
- `base_hsv` or at least `base_v` should be computed once per mesh from `base_rgb`.
- If optional layer delays are enabled, compute one scalar delay per mesh at scene-build time.

Memory layout considerations:

- Keep point arrays and color arrays as `float32` where possible, matching the current `pv.PolyData(np.asarray(points, dtype=np.float32))` in [visualization/animation.py](visualization/animation.py).
- `projected_s` should preferably be `float32` as well unless numeric instability appears at very large coordinate scales.
- Per-frame temporary arrays should ideally be one `(N,)` brightness vector and one `(N,3)` RGB array per mesh.
- Avoid storing redundant per-point HSV arrays. They are not needed if you use the RGB scaling identity described above.

Avoiding Python loops:

- Python loops over layers are fine because there are only up to nine layers.
- Python loops over individual points are not acceptable for the frame path.
- Do not call `colorsys.rgb_to_hsv` or `colorsys.hsv_to_rgb` per point.

Expected computational cost per frame:

- For each mesh with `N` points, the dominant work is:
  one vectorized sine over `N`,
  one vectorized clip over `N`,
  one broadcasted multiply over `(N,3)`,
  one assignment into the color buffer.
- Complexity is $O(N)$ per mesh per frame.
- With nine layers, total cost is $O(\sum N_\ell)$ per frame.
- That is the right complexity class for this visualization and should remain practical unless the dataset becomes very large.

**7. Optional Layer Coupling Model**
The Phase 4 spec explicitly allows a simple first-pass coupling model. It should remain a delay model, not a full network model.

Recommended v1 coupling semantics:

- Interpret `config.layer_coupling` as delay per layer step, in seconds.
- For a displayed layer list `[1, 2, 3, ...]`, compute:
  $$
  \Delta t_\ell = (\ell - \ell_{ref}) \cdot layer\_coupling
  $$
  where $\ell_{ref}$ is either the minimum displayed layer number or the first layer in the input order.
- Then the layer brightness becomes:
  $$
  B_\ell(\mathbf{x}, t)=1 + A\sin(s-\omega t+\phi+\omega\Delta t_\ell)
  $$

Why this model is safe:

- It does not change the current Phase 3 branch.
- It does not require any topology matrix or neuron-level connectivity data.
- It is easy to reason about and test.
- Setting `layer_coupling = 0.0` exactly disables it.

How to integrate delays without breaking Phase 3:

- Only compute or apply per-layer delays when `mode == "traveling_wave"`.
- Keep `global_pulse` independent of layer delays unless you intentionally introduce a separate future mode for delayed global pulse.

How to expose coupling parameters safely:

- Keep only the single scalar `layer_coupling` in `AnimationConfig` for v1.
- Document it as delay-per-layer-step, not “influence strength,” because the latter is vague and underspecified mathematically.
- If richer coupling is needed later, defer it to a later sub-phase rather than overloading this field with multiple meanings.

**8. CLI Integration For Manual Testing**
The manual harness in phase3_manualtest.py is the right place to expose Phase 4 controls without touching public viewer APIs.

New CLI flags:

- `--mode {global_pulse,traveling_wave}`
  Maps directly to `AnimationConfig.mode`.
- `--wave-direction X Y Z`
  Maps to `AnimationConfig.wave_direction`.
- `--wavelength FLOAT`
  Maps to `AnimationConfig.wavelength`.
- `--velocity FLOAT`
  Maps to `AnimationConfig.velocity`.
- `--phase-offset FLOAT`
  Maps to `AnimationConfig.phase_offset`.
- `--layer-coupling FLOAT`
  Maps to `AnimationConfig.layer_coupling`.

Mapping behavior:

- In `global_pulse` mode:
  use existing `--amplitude` and `--frequency`,
  ignore `wave-direction`, `wavelength`, `velocity`, `phase-offset`, `layer-coupling`.
- In `traveling_wave` mode:
  use `--amplitude`, `--wave-direction`, `--wavelength`, `--velocity`, `--phase-offset`, `--layer-coupling`,
  ignore `--frequency` for brightness timing because the wave speed is derived from `velocity` and `wavelength`.
- The HUD already added to the manual script should display mode and enough wave parameters to verify the branch in use.

Validation rules for CLI parsing:

- `wave_direction` cannot be the zero vector.
- `wavelength` must be `> 0`.
- `velocity` can be `0` if you want a standing spatial phase pattern frozen in time.
- `layer_coupling` should default to `0.0`.
- If `mode == "traveling_wave"` and required wave parameters are invalid, the script should fail before opening the viewer.

**9. Testing And Validation Plan**
Phase 4 needs both math tests and regression protection for current Phase 3 behavior in phase3_test.py.

Unit tests for brightness computation:

- Test that `mode="global_pulse"` still reproduces the current Phase 3 scalar brightness behavior.
- Test that `mode="traveling_wave"` produces different brightness values for two points with different projected positions at the same time.
- Test that points with identical projected `s` values produce identical brightness at the same time.
- Test that reversing `wave_direction` reverses spatial phase ordering.
- Test that `velocity=0` produces a spatial pattern that does not change over time.
- Test that `phase_offset` shifts the waveform but does not change amplitude bounds.
- Test that `layer_coupling=0` yields identical timing across layers, all else equal.

Regression tests for Phase 3:

- Keep all existing Phase 3 tests passing unchanged for the default config.
- Add an explicit test asserting that `AnimationConfig()` defaults to `mode="global_pulse"` once the field is added.
- Add one regression test that `generate_frame` in `global_pulse` mode still produces uniform per-layer color arrays, meaning every point in a layer has the same RGB value after a frame update.

Visual tests for traveling waves:

- Single layer, long wavelength, simple direction `(1, 0, 0)`, moderate velocity.
- Single layer, oblique direction `(1, 1, 0)` normalized, to confirm the wavefront is not axis-locked.
- Two or more layers with nonzero `layer_coupling` to verify delayed phase progression.
- Compare `global_pulse` vs `traveling_wave` in the manual viewer using the same amplitude and duration.

Edge-case tests:

- `amplitude=0`: no brightness change in either mode.
- `velocity=0`: static spatial phase field.
- Very large wavelength: should visually approach a global pulse because spatial variation across the scene becomes small.
- Very small wavelength: rapid spatial oscillation, still no exceptions.
- Zero or invalid `wave_direction`: should raise a validation error.
- Zero or negative `wavelength`: should raise a validation error.
- Unsupported `mode`: should raise a `ValueError`.
- Future-proof safety case: if a base color ever has `v=0`, the code should avoid division-by-zero in RGB scaling.

**10. Implementation Roadmap**
This should be implemented in a sequence that protects current behavior and keeps each step testable.

1. Extend `AnimationConfig` in animation.py.
- Add `mode`, `wave_direction`, `wavelength`, `velocity`, `phase_offset`, `layer_coupling`.
- Keep defaults backward-compatible with Phase 3.
- Test immediately: config construction, defaults, and serialization-like inspection.

2. Extend the private animation scene cache in `_build_animation_scene`.
- Keep the existing mesh creation and `mesh.point_data["colors"]` behavior unchanged.
- Add cached layer metadata needed for Phase 4: `base_v`, layer number, and placeholders for projected coordinates and delay.
- Test immediately: scene still builds; existing Phase 3 tests remain green.

3. Add internal wave-parameter preparation in animation.py.
- Compute normalized direction, `k`, and `omega`.
- Compute per-layer delay scalars if `layer_coupling != 0`.
- Precompute `projected_s` for each mesh.
- Test immediately: deterministic values for simple inputs.

4. Integrate exact mode dispatch into `generate_frame`.
- Leave the Phase 3 branch intact.
- Add the traveling-wave branch.
- Ensure `plotter.render()` and `mesh.Modified()` semantics remain unchanged.
- Test immediately: new unit tests for brightness-by-position and regression tests for `global_pulse`.

5. Optimize the traveling-wave color path.
- Use vectorized RGB scaling from precomputed base color and base value, not per-point `colorsys`.
- Verify that hue preservation remains true numerically.
- Test immediately: hue/saturation invariants and timing behavior.

6. Keep `play_animation` and `export_animation` signatures untouched.
- Confirm they require no signature changes because both already delegate to `generate_frame`.
- Test immediately: smoke tests for off-screen playback and MP4 export in both modes.

7. Extend phase3_manualtest.py CLI.
- Add the Phase 4 flags listed above.
- Surface mode and wave parameters in the HUD.
- Test immediately: parse validation and visual inspection runs.

8. Add optional `layer_coupling` behavior only after the base traveling wave is stable.
- This should be a second pass, not part of the first minimal traveling-wave delivery.
- Test immediately: delayed phase between layers, zero-coupling no-op behavior.

9. Defer nonessential sub-phases.
- Defer non-sinusoidal waveforms.
- Defer connectivity-informed coupling.
- Defer richer coupling matrices.
- Defer any viewer UI beyond the manual test script.
- Defer any refactor of viewer.py, because it is not needed for Phase 4.
