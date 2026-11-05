# i02 terminal failure analysis

This analysis uses only saved artifacts from the **completed** `ALFRED_V9_20261105_pilot15_i02` episode. No model, GPU, simulator or mapping replay was run, and no findings were fed back to any active actor. Full source paths and SHA-256 values are in `i02_terminal_failure_cases.json`.

The episode ended with **0/3 goal conditions**, 33 actions, 38 decisions and the 10-failure budget exhausted. Every failed action was `OpenObject`. There were only two `MoveAhead` actions and three distinct base positions. Of 15 navigation requests, 14 returned either no reachable stance or no safe next 0.25 m primitive; one reported arrival while explicitly leaving interaction readiness unverified.

## Case 1: fridge identity changes and unverified approach

Actions **18, 20, 25 and 27** addressed memory IDs **88, 4, 122 and 129**. The saved official action records resolve all four to the same fridge; the report preserves a hash of that simulator identity. All four use a mask from their current source frame, so stale historical mask reuse is not supported by these records.

In **frame 19 / decision 21 / action 20**, the raw fridge mask contains **22,203 pixels**, while the executed mapping mask contains **15,284 pixels**. Visual inspection of the original RGB shows that a retained `Footstool` detection covers the lower fridge compartment, while `Curtains` detections cover the handles. Those contained masks overlap **5,440** and **1,479** removed pixels respectively; a second 737-pixel Curtains overlap is nested and must not be added again. The original detector box is `[151.9998, 23.2460, 271.5863, 241.7483]` with confidence **0.536**.

The recorded source of the interaction is `current_mapping_mask`. The source code explicitly subtracts contained object masks for both geometry and interactions. This verifies a concrete change in the action mask, but **does not establish that subtraction caused the failure**: the official interaction still selected the fridge. The predicted depth median on the executed mask is **2.206 m**, and a later retry followed a navigation result saying no safe primitive was available. These support investigating approach readiness; predicted depth is not GT range, and no counterfactual approach was tested.

## Case 2: Cabinet memory, Shelf detection, Microwave selection

The final **decision 38 / action 33 / frame 32** requests opening memory object **68**, labeled `Cabinet`. Its current detection is **index 9, Shelf**, with confidence **0.469** and box `[0.0347, 86.1958, 78.3571, 157.9234]`. The memory category evidence is `Cabinet: 3, Shelf: 2`. The RGB shows this broad region at the left edge spanning the upper cupboard and microwave area.

The raw mask has **5,395 pixels**, but the executed mask has **1,115 pixels**, largely a narrow remaining outline. Removed regions include the **1,844-pixel Microwave** mask and a **2,434-pixel smaller Shelf** mask. The official saved action selects **Microwave**, while the actor is reasoning about a cabinet. The immediately preceding navigation request reported no reachable stance.

**Evaluator-only, read after terminal:** because this is the final action and no later movement occurs, the terminal snapshot directly applies to its pose. The selected microwave is openable and closed, with `visible=false` and recorded distance **3.69716787 m**. These hidden fields were never actor inputs. They support lack of interaction readiness for this final attempt; `visible=false` does not mean the object was absent from the RGB. The evidence shows disagreement between remembered category, current region and selected interaction object, but does not isolate one upstream algorithm as the sole cause.

## Timing limits

Recorded run wall time is **1,947.11 s**. Model-policy calls account for **1,526.57 s (78.4%)**, and perception RPCs for **211.64 s**. Worker inference is **116.63 s** over 34 frames; its **114.39 s** GPU-lock wait includes initialization and overlaps other timing scopes. These timers must not be summed as independent components.

The approximately **208.90 s** remainder includes initialization, simulator actions, CPU mapping/navigation, rendering and I/O. It is **not measured mapping CPU time**. The code's cKDTree neighborhood lists and Python clustering loops are possible scaling costs; the 167 final memory objects and 936,976 input tokens also make context growth a concrete concern. No live-process profiling was performed.

Saved visualization artifacts:

- `i02_saved_mask_comparison.png`: original RGB, original SAM mask and executed mask for frames 19 and 32.
- `i02_visual_timeline.png`: four stages using only the saved current RGB and BEV.
- `i02_visual_timeline_sources.json`: source-panel paths and hashes.

The same storyboard format can be applied to i03 after it reaches terminal state, selecting its first successful pickup, the immediately updated held-object memory, a subsequent navigation/interaction, and terminal output. Its GT is not required for preparing those actor-visible panels.
