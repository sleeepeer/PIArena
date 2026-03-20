---
title: Defenses
slug: defenses
category: defenses
---

# Defenses

PIArena supports several defense styles:

- detection defenses that block or warn on suspicious context
- sanitization defenses that remove suspicious content before querying the model
- hybrid defenses that detect and then recover a cleaned version of the context

## Supported Defenses

- [`none`](/docs/defenses/none): no defense
- [`pisanitizer`](/docs/defenses/pisanitizer): attention-based sanitization
- [`secalign`](/docs/defenses/secalign): alignment model used directly as the answering model
- [`datasentinel`](/docs/defenses/datasentinel): detector that blocks suspicious context
- [`attentiontracker`](/docs/defenses/attentiontracker): attention-pattern detector
- [`promptguard`](/docs/defenses/promptguard): classifier-based detector
- [`promptlocate`](/docs/defenses/promptlocate): detect, localize, and recover
- [`promptarmor`](/docs/defenses/promptarmor): auxiliary-model detection and cleanup
- [`datafilter`](/docs/defenses/datafilter): recursive content filtering
- [`piguard`](/docs/defenses/piguard): fine-tuned prompt injection classifier

## How To Choose

- Start with `none` when you want a baseline.
- Try `promptguard`, `datasentinel`, `attentiontracker`, or `piguard` when you want detection behavior.
- Try `pisanitizer`, `datafilter`, `promptarmor`, or `promptlocate` when you want cleaned context instead of a warning-only block.
- Use `secalign` when you want to run a specialized aligned model as the defense path itself.
